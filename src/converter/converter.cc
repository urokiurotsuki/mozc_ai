// Copyright 2010-2021, Google Inc.
// All rights reserved.
//
// Mozc AI IME v7.0 - High-Performance Japanese Input System
//
// 設計思想:
// - レガシー対応を廃止し、将来の機能拡張に特化
// - V2プロトコルによる拡張性の確保
// - 堅牢なエラーハンドリング
// - 高い応答性とリアルタイム性
//
// 主要機能:
// 1. CONVERT: 文脈全体からの予測変換
// 2. TAB: 英語変換 + 誤字脱字修正
// 3. LEARN: 自律的学習
//
// プロトコル V2:
//   V2\t{mode}\t{options}\t{history}\t{seg_count}\t{reading1}\t{cand_count1}\t{cand1}...
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of Google Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "converter/converter.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#include <sstream>
#include <chrono>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <atomic>
#endif

#include "absl/base/optimization.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/random/random.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "base/strings/assign.h"
#include "base/util.h"
#include "base/vlog.h"
#include "composer/composer.h"
#include "converter/attribute.h"
#include "converter/candidate.h"
#include "converter/history_reconstructor.h"
#include "converter/immutable_converter_interface.h"
#include "converter/inner_segment.h"
#include "converter/reverse_converter.h"
#include "converter/segments.h"
#include "dictionary/pos_matcher.h"
#include "engine/modules.h"
#include "prediction/predictor_interface.h"
#include "prediction/result.h"
#include "protocol/commands.pb.h"
#include "request/conversion_request.h"
#include "rewriter/rewriter_interface.h"
#include "transliteration/transliteration.h"

namespace mozc {
namespace converter {
namespace {

// ==========================================
// AI Server Client v7.0 - Future-Proof Design
// ==========================================

#ifdef _WIN32

// ===========================================
// 定数定義
// ===========================================

// プロトコルバージョン
static constexpr const char* kProtocolVersion = "V2";

// 変換モード
static constexpr const char* kModeConvert = "CONVERT";
static constexpr const char* kModeTab = "TAB";
static constexpr const char* kModeLearn = "LEARN";

// パイプ名
static constexpr const char* kAiPipeName = "\\\\.\\pipe\\MozcAIPipe";
static constexpr const char* kAiLearnPipeName = "\\\\.\\pipe\\MozcAILearnPipe";

// タイムアウト設定（ミリ秒）
static constexpr DWORD kPipeTimeoutMs = 80;
static constexpr DWORD kTabPipeTimeoutMs = 150;  // TAB補完は少し長め
static constexpr DWORD kLearnPipeTimeoutMs = 30;

// 候補数制限
static constexpr int kMaxCandidatesPerSegment = 6;
static constexpr int kMaxTotalCandidates = 30;

// バッファサイズ
static constexpr size_t kPipeBufferSize = 65536;
static constexpr size_t kDebugBufferSize = 4096;

// ===========================================
// エラーコード
// ===========================================

enum class AiError {
    kSuccess = 0,
    kPipeConnectionFailed,
    kPipeWriteFailed,
    kPipeReadFailed,
    kTimeout,
    kInvalidResponse,
    kEmptySegments,
    kUnknownError
};

// ===========================================
// 統計情報（デバッグ用）
// ===========================================

struct AiStatistics {
    std::atomic<uint64_t> total_requests{0};
    std::atomic<uint64_t> successful_requests{0};
    std::atomic<uint64_t> failed_requests{0};
    std::atomic<uint64_t> total_latency_us{0};
    std::atomic<uint64_t> cache_hits{0};
    std::mutex stats_mutex;

    void RecordRequest(bool success, uint64_t latency_us) {
        total_requests++;
        if (success) {
            successful_requests++;
        } else {
            failed_requests++;
        }
        total_latency_us += latency_us;
    }

    double GetAverageLatencyMs() const {
        uint64_t total = total_requests.load();
        if (total == 0) return 0.0;
        return static_cast<double>(total_latency_us.load()) / total / 1000.0;
    }
};

static AiStatistics g_ai_stats;

// ===========================================
// デバッグログ
// ===========================================

inline void DebugLog(const char* format, ...) {
    char buffer[kDebugBufferSize];
    va_list args;
    va_start(args, format);
    vsnprintf(buffer, sizeof(buffer), format, args);
    va_end(args);
    ::OutputDebugStringA(buffer);
}

// ===========================================
// ユーティリティ関数
// ===========================================

/**
 * 現在時刻をマイクロ秒で取得
 */
inline uint64_t GetCurrentTimeUs() {
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
}

/**
 * 文字列をエスケープ（タブと改行を置換）
 */
std::string EscapeString(const std::string& input) {
    std::string result;
    result.reserve(input.size());
    for (char c : input) {
        if (c == '\t') {
            result += "\\t";
        } else if (c == '\n') {
            result += "\\n";
        } else if (c == '\r') {
            result += "\\r";
        } else {
            result += c;
        }
    }
    return result;
}

/**
 * 履歴コンテキストを取得
 * 直前に確定した文字列を連結して返す
 */
std::string GetHistoryContext(const Segments& segments, size_t max_chars = 300) {
    std::string history;
    size_t char_count = 0;
    
    // 逆順で走査して最新の履歴から取得
    for (int i = static_cast<int>(segments.history_segments_size()) - 1; i >= 0; --i) {
        const Segment& seg = segments.history_segment(i);
        if (seg.candidates_size() > 0) {
            const std::string& value = seg.candidate(0).value;
            size_t value_len = Util::CharsLen(value);
            
            if (char_count + value_len > max_chars) {
                break;
            }
            
            history = value + history;
            char_count += value_len;
        }
    }
    
    return history;
}

/**
 * デフォルトオプションJSON
 */
std::string GetDefaultOptions() {
    return "{}";
}

// ===========================================
// パイプ通信クラス
// ===========================================

class PipeClient {
public:
    /**
     * パイプにリクエストを送信してレスポンスを取得
     */
    static AiError SendRequest(
        const char* pipe_name,
        const std::string& request,
        std::string& response,
        DWORD timeout_ms
    ) {
        if (request.empty()) {
            return AiError::kEmptySegments;
        }

        char buffer[kPipeBufferSize];
        DWORD bytes_read = 0;

        BOOL success = ::CallNamedPipeA(
            pipe_name,
            const_cast<char*>(request.c_str()),
            static_cast<DWORD>(request.size()),
            buffer,
            sizeof(buffer),
            &bytes_read,
            timeout_ms
        );

        if (!success) {
            DWORD error = ::GetLastError();
            if (error == ERROR_PIPE_BUSY) {
                return AiError::kTimeout;
            } else if (error == ERROR_FILE_NOT_FOUND) {
                return AiError::kPipeConnectionFailed;
            }
            return AiError::kUnknownError;
        }

        if (bytes_read == 0) {
            return AiError::kPipeReadFailed;
        }

        response.assign(buffer, bytes_read);
        return AiError::kSuccess;
    }

    /**
     * 非同期でリクエストを送信（学習通知用）
     */
    static void SendRequestAsync(
        const char* pipe_name,
        const std::string& request,
        DWORD timeout_ms
    ) {
        // 簡易的な非同期処理（別スレッドで実行）
        std::string request_copy = request;
        std::thread([pipe_name, request_copy, timeout_ms]() {
            char buffer[256];
            DWORD bytes_read = 0;
            ::CallNamedPipeA(
                pipe_name,
                const_cast<char*>(request_copy.c_str()),
                static_cast<DWORD>(request_copy.size()),
                buffer,
                sizeof(buffer),
                &bytes_read,
                timeout_ms
            );
        }).detach();
    }
};

// ===========================================
// プロトコルビルダー
// ===========================================

class ProtocolBuilder {
public:
    /**
     * CONVERT リクエストを構築
     * フォーマット: V2\tCONVERT\t{options}\t{history}\t{seg_count}\t{reading1}\t{cand_count1}\t{cand1}...
     */
    static std::string BuildConvertRequest(
        const std::string& history,
        const Segments& segments,
        const std::string& options = "{}"
    ) {
        std::ostringstream payload;
        
        size_t seg_count = segments.conversion_segments_size();
        if (seg_count == 0) {
            return "";
        }

        // ヘッダー
        payload << kProtocolVersion << "\t"
                << kModeConvert << "\t"
                << options << "\t"
                << EscapeString(history) << "\t"
                << seg_count;

        // 各セグメントのデータ
        for (size_t i = 0; i < seg_count; ++i) {
            const Segment& seg = segments.conversion_segment(i);
            std::string reading(seg.key().data(), seg.key().size());

            int cand_count = std::min(
                static_cast<int>(seg.candidates_size()),
                kMaxCandidatesPerSegment
            );

            payload << "\t" << EscapeString(reading);
            payload << "\t" << cand_count;

            for (int j = 0; j < cand_count; ++j) {
                payload << "\t" << EscapeString(seg.candidate(j).value);
            }
        }

        return payload.str();
    }

    /**
     * TAB補完リクエストを構築
     * フォーマット: V2\tTAB\t{options}\t{history}\t{raw_input}
     */
    static std::string BuildTabRequest(
        const std::string& history,
        const std::string& raw_input,
        const std::string& options = "{}"
    ) {
        std::ostringstream payload;
        
        payload << kProtocolVersion << "\t"
                << kModeTab << "\t"
                << options << "\t"
                << EscapeString(history) << "\t"
                << EscapeString(raw_input);

        return payload.str();
    }

    /**
     * 学習通知リクエストを構築
     * フォーマット: LEARN\t{reading1}\t{proposed1}\t{committed1}...
     */
    static std::string BuildLearnRequest(
        const std::vector<std::tuple<std::string, std::string, std::string>>& learn_data
    ) {
        if (learn_data.empty()) {
            return "";
        }

        std::ostringstream payload;
        payload << kModeLearn;

        for (const auto& entry : learn_data) {
            payload << "\t" << EscapeString(std::get<0>(entry))
                    << "\t" << EscapeString(std::get<1>(entry))
                    << "\t" << EscapeString(std::get<2>(entry));
        }

        return payload.str();
    }
};

// ===========================================
// レスポンスパーサー
// ===========================================

class ResponseParser {
public:
    /**
     * CONVERTレスポンスを解析
     * フォーマット: {selected1}\t{selected2}...
     */
    static std::vector<std::string> ParseConvertResponse(const std::string& response) {
        std::vector<std::string> results;
        
        if (response.empty()) {
            return results;
        }

        std::istringstream iss(response);
        std::string token;
        
        while (std::getline(iss, token, '\t')) {
            // 末尾の改行を除去
            while (!token.empty() && (token.back() == '\n' || token.back() == '\r')) {
                token.pop_back();
            }
            results.push_back(token);
        }

        return results;
    }

    /**
     * TAB補完レスポンスを解析
     * フォーマット: {suggestion1}\t{suggestion2}...\t{typo_fix}
     */
    static std::pair<std::vector<std::string>, std::string> ParseTabResponse(
        const std::string& response
    ) {
        std::vector<std::string> suggestions;
        std::string typo_fix;

        if (response.empty()) {
            return {suggestions, typo_fix};
        }

        std::istringstream iss(response);
        std::string token;
        bool first = true;

        while (std::getline(iss, token, '\t')) {
            // 末尾の改行を除去
            while (!token.empty() && (token.back() == '\n' || token.back() == '\r')) {
                token.pop_back();
            }

            if (first && token.size() > 5 && token.substr(0, 5) == "TYPO:") {
                typo_fix = token.substr(5);
                first = false;
            } else {
                suggestions.push_back(token);
            }
        }

        return {suggestions, typo_fix};
    }
};

// ===========================================
// 学習データトラッカー
// ===========================================

class LearnTracker {
public:
    struct ProposalEntry {
        std::string reading;
        std::string proposed_value;
        uint64_t timestamp;
    };

    /**
     * AIの提案を記録
     */
    void RecordProposal(const std::string& reading, const std::string& value) {
        std::lock_guard<std::mutex> lock(mutex_);
        proposals_[reading] = ProposalEntry{
            reading,
            value,
            GetCurrentTimeUs()
        };
        
        // 古いエントリをクリーンアップ（5秒以上前）
        CleanupOldEntries();
    }

    /**
     * 確定時に提案と比較して学習データを生成
     */
    std::vector<std::tuple<std::string, std::string, std::string>> 
    GenerateLearnData(const Segments& segments) {
        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<std::tuple<std::string, std::string, std::string>> learn_data;

        for (size_t i = 0; i < segments.conversion_segments_size(); ++i) {
            const Segment& seg = segments.conversion_segment(i);
            if (seg.candidates_size() == 0) continue;

            std::string reading(seg.key().data(), seg.key().size());
            std::string committed = seg.candidate(0).value;

            // 正規化
            std::string normalized_reading = NormalizeReading(reading);

            auto it = proposals_.find(normalized_reading);
            if (it != proposals_.end()) {
                // 提案と確定が異なる場合のみ学習
                if (it->second.proposed_value != committed) {
                    learn_data.emplace_back(
                        normalized_reading,
                        it->second.proposed_value,
                        committed
                    );
                    DebugLog("[MozcAI] Learn: %s | proposed=%s -> committed=%s\n",
                             reading.c_str(),
                             it->second.proposed_value.c_str(),
                             committed.c_str());
                }
                proposals_.erase(it);
            } else {
                // 提案記録がない場合は空の提案として記録
                learn_data.emplace_back(normalized_reading, "", committed);
            }
        }

        return learn_data;
    }

    /**
     * 提案をクリア
     */
    void Clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        proposals_.clear();
    }

private:
    /**
     * 読みを正規化（長音記号などを除去）
     * UTF-8文字列として処理
     */
    std::string NormalizeReading(const std::string& reading) {
        std::string result;
        result.reserve(reading.size());
        
        size_t i = 0;
        while (i < reading.size()) {
            unsigned char c = static_cast<unsigned char>(reading[i]);
            
            // UTF-8のバイト数を判定
            size_t char_len = 1;
            if ((c & 0x80) == 0) {
                // ASCII (1 byte)
                char_len = 1;
            } else if ((c & 0xE0) == 0xC0) {
                // 2 bytes
                char_len = 2;
            } else if ((c & 0xF0) == 0xE0) {
                // 3 bytes (日本語はここ)
                char_len = 3;
            } else if ((c & 0xF8) == 0xF0) {
                // 4 bytes
                char_len = 4;
            }
            
            // 文字を抽出
            if (i + char_len <= reading.size()) {
                std::string ch = reading.substr(i, char_len);
                
                // 長音記号「ー」(U+30FC: E3 83 BC) と中点「・」(U+30FB: E3 83 BB) を除外
                bool skip = false;
                if (char_len == 3) {
                    if (ch == "\xE3\x83\xBC" || ch == "\xE3\x83\xBB") {
                        skip = true;
                    }
                }
                
                if (!skip) {
                    result += ch;
                }
            }
            
            i += char_len;
        }
        
        return result;
    }

    void CleanupOldEntries() {
        uint64_t now = GetCurrentTimeUs();
        uint64_t threshold = 5000000; // 5秒

        for (auto it = proposals_.begin(); it != proposals_.end();) {
            if (now - it->second.timestamp > threshold) {
                it = proposals_.erase(it);
            } else {
                ++it;
            }
        }
    }

    std::mutex mutex_;
    std::unordered_map<std::string, ProposalEntry> proposals_;
};

static LearnTracker g_learn_tracker;

// ===========================================
// AI変換クライアント
// ===========================================

/**
 * AIサーバーに全文節を一括送信してリランキング
 */
bool QueryAiConversionBatch(const std::string& history_context, 
                            Segments* segments) {
    if (segments->conversion_segments_size() == 0) {
        return false;
    }

    uint64_t start_time = GetCurrentTimeUs();
    size_t seg_count = segments->conversion_segments_size();

    // リクエスト構築
    std::string request = ProtocolBuilder::BuildConvertRequest(
        history_context,
        *segments,
        GetDefaultOptions()
    );

    if (request.empty()) {
        DebugLog("[MozcAI] Empty request, skipping\n");
        return false;
    }

    // デバッグログ
    std::ostringstream debug_send;
    debug_send << "[MozcAI] >>> V2 CONVERT hist:" << history_context.length()
               << " segs:" << seg_count << " [";
    for (size_t i = 0; i < std::min(seg_count, size_t(3)); ++i) {
        const Segment& seg = segments->conversion_segment(i);
        std::string reading(seg.key().data(), seg.key().size());
        debug_send << reading << "(";
        for (size_t j = 0; j < std::min(seg.candidates_size(), size_t(2)); ++j) {
            if (j > 0) debug_send << "/";
            debug_send << seg.candidate(j).value;
        }
        debug_send << ")";
        if (i < seg_count - 1 && i < 2) debug_send << " ";
    }
    if (seg_count > 3) debug_send << " ...";
    debug_send << "]\n";
    DebugLog("%s", debug_send.str().c_str());

    // パイプ通信
    std::string response;
    AiError error = PipeClient::SendRequest(
        kAiPipeName,
        request,
        response,
        kPipeTimeoutMs
    );

    if (error != AiError::kSuccess) {
        uint64_t latency = GetCurrentTimeUs() - start_time;
        g_ai_stats.RecordRequest(false, latency);
        DebugLog("[MozcAI] Pipe error: %d (latency=%llums)\n", 
                 static_cast<int>(error), latency / 1000);
        return false;
    }

    // レスポンス解析
    std::vector<std::string> selected_values = ResponseParser::ParseConvertResponse(response);

    // デバッグ出力
    std::ostringstream debug_recv;
    debug_recv << "[MozcAI] <<< ";
    for (size_t i = 0; i < selected_values.size(); ++i) {
        debug_recv << "[" << selected_values[i] << "]";
    }
    debug_recv << "\n";
    DebugLog("%s", debug_recv.str().c_str());

    // 結果適用
    size_t applied_count = 0;
    size_t same_count = 0;

    for (size_t i = 0; i < seg_count && i < selected_values.size(); ++i) {
        const std::string& selected = selected_values[i];
        if (selected.empty()) {
            continue;
        }

        Segment* seg = segments->mutable_conversion_segment(i);
        std::string reading(seg->key().data(), seg->key().size());

        // 学習トラッカーに記録
        g_learn_tracker.RecordProposal(reading, selected);

        // 選択された候補を探す
        int found_index = -1;
        for (size_t j = 0; j < seg->candidates_size(); ++j) {
            if (seg->candidate(j).value == selected) {
                found_index = static_cast<int>(j);
                break;
            }
        }

        if (found_index > 0) {
            // 候補を先頭に移動
            seg->move_candidate(found_index, 0);
            seg->mutable_candidate(0)->attributes |= Attribute::RERANKED;
            applied_count++;
        } else if (found_index == 0) {
            // 既に先頭
            same_count++;
        } else {
            // AI生成の新候補を追加
            Candidate* cand = seg->push_back_candidate();
            cand->key = reading;
            cand->content_key = reading;
            cand->value = selected;
            cand->content_value = selected;
            cand->cost = 0;
            cand->structure_cost = 0;

            if (seg->candidates_size() > 1) {
                const Candidate& first = seg->candidate(0);
                cand->lid = first.lid;
                cand->rid = first.rid;
            } else {
                cand->lid = 0;
                cand->rid = 0;
            }
            cand->attributes |= Attribute::RERANKED;

            // 先頭に移動
            if (seg->candidates_size() > 1) {
                seg->move_candidate(seg->candidates_size() - 1, 0);
            }
            applied_count++;
        }
    }

    uint64_t latency = GetCurrentTimeUs() - start_time;
    g_ai_stats.RecordRequest(true, latency);
    
    DebugLog("[MozcAI] Result: applied=%zu same=%zu total=%zu (latency=%llums)\n",
             applied_count, same_count, seg_count, latency / 1000);

    return applied_count > 0 || same_count > 0;
}

/**
 * TAB補完でAIに問い合わせ
 * 英語変換と誤字脱字修正を取得
 */
bool QueryAiTabCompletion(const std::string& history_context,
                          const std::string& raw_input,
                          Segment* segment) {
    if (raw_input.empty() || segment == nullptr) {
        return false;
    }

    uint64_t start_time = GetCurrentTimeUs();

    // リクエスト構築
    std::string options = R"({"include_english":true,"include_typo_fix":true})";
    std::string request = ProtocolBuilder::BuildTabRequest(
        history_context,
        raw_input,
        options
    );

    DebugLog("[MozcAI] >>> V2 TAB input=%s\n", raw_input.c_str());

    // パイプ通信
    std::string response;
    AiError error = PipeClient::SendRequest(
        kAiPipeName,
        request,
        response,
        kTabPipeTimeoutMs
    );

    if (error != AiError::kSuccess) {
        DebugLog("[MozcAI] TAB pipe error: %d\n", static_cast<int>(error));
        return false;
    }

    // レスポンス解析
    auto [suggestions, typo_fix] = ResponseParser::ParseTabResponse(response);

    DebugLog("[MozcAI] <<< TAB suggestions=%zu typo_fix=%s\n",
             suggestions.size(), typo_fix.c_str());

    // 候補を追加
    bool added = false;

    // 誤字脱字修正を最優先で追加
    if (!typo_fix.empty()) {
        Candidate* cand = segment->push_back_candidate();
        cand->key = raw_input;
        cand->content_key = raw_input;
        cand->value = typo_fix;
        cand->content_value = typo_fix;
        cand->description = "修正候補";
        cand->cost = -10000;  // 最優先
        cand->attributes |= Attribute::RERANKED;
        
        // 先頭に移動
        if (segment->candidates_size() > 1) {
            segment->move_candidate(segment->candidates_size() - 1, 0);
        }
        added = true;
    }

    // 英語候補を追加
    for (const auto& suggestion : suggestions) {
        if (suggestion.empty()) continue;

        // 重複チェック
        bool exists = false;
        for (size_t i = 0; i < segment->candidates_size(); ++i) {
            if (segment->candidate(i).value == suggestion) {
                exists = true;
                break;
            }
        }

        if (!exists) {
            Candidate* cand = segment->push_back_candidate();
            cand->key = raw_input;
            cand->content_key = raw_input;
            cand->value = suggestion;
            cand->content_value = suggestion;
            cand->description = "英語";
            cand->cost = -5000;
            cand->attributes |= Attribute::RERANKED;
            added = true;
        }
    }

    uint64_t latency = GetCurrentTimeUs() - start_time;
    DebugLog("[MozcAI] TAB result: added=%d (latency=%llums)\n", added, latency / 1000);

    return added;
}

/**
 * 確定情報をAIサーバーに通知（学習用）
 */
void NotifyLearnToAI(const Segments& segments) {
    // 学習データを生成
    auto learn_data = g_learn_tracker.GenerateLearnData(segments);

    if (learn_data.empty()) {
        return;
    }

    // リクエスト構築
    std::string request = ProtocolBuilder::BuildLearnRequest(learn_data);

    if (request.empty()) {
        return;
    }

    DebugLog("[MozcAI] Learn notify: %zu entries\n", learn_data.size());

    // 非同期で送信（メインスレッドをブロックしない）
    PipeClient::SendRequestAsync(kAiLearnPipeName, request, kLearnPipeTimeoutMs);
}

#else
// Non-Windows stubs
inline void DebugLog(const char* format, ...) {}
std::string GetHistoryContext(const Segments& segments, size_t max_chars = 300) { return ""; }
bool QueryAiConversionBatch(const std::string& history_context, Segments* segments) { return false; }
bool QueryAiTabCompletion(const std::string& history_context, const std::string& raw_input, Segment* segment) { return false; }
void NotifyLearnToAI(const Segments& segments) {}

class LearnTracker {
public:
    void RecordProposal(const std::string&, const std::string&) {}
    void Clear() {}
};
static LearnTracker g_learn_tracker;
#endif

// ==========================================
// End of AI Server Client
// ==========================================

constexpr size_t kErrorIndex = static_cast<size_t>(-1);

size_t GetSegmentIndex(const Segments* segments, size_t segment_index) {
  const size_t history_segments_size = segments->history_segments_size();
  const size_t result = history_segments_size + segment_index;
  if (result >= segments->segments_size()) {
    return kErrorIndex;
  }
  return result;
}

bool ShouldInitSegmentsForPrediction(absl::string_view key,
                                     const Segments& segments) {
  return segments.conversion_segments_size() == 0 ||
         segments.conversion_segment(0).key() != key;
}

bool IsValidSegments(const ConversionRequest& request,
                     const Segments& segments) {
  const bool is_mobile = request.request().zero_query_suggestion() &&
                         request.request().mixed_conversion();

  for (const Segment& segment : segments) {
    if (segment.candidates_size() != 0) {
      continue;
    }
    if (is_mobile && segment.meta_candidates_size() != 0) {
      continue;
    }
    return false;
  }
  return true;
}

}  // namespace

Converter::Converter(
    std::unique_ptr<engine::Modules> modules,
    const ImmutableConverterFactory& immutable_converter_factory,
    const PredictorFactory& predictor_factory,
    const RewriterFactory& rewriter_factory)
    : modules_(std::move(modules)),
      immutable_converter_(immutable_converter_factory(*modules_)),
      pos_matcher_(modules_->GetPosMatcher()),
      user_dictionary_(modules_->GetUserDictionary()),
      history_reconstructor_(modules_->GetPosMatcher()),
      reverse_converter_(*immutable_converter_),
      general_noun_id_(pos_matcher_.GetGeneralNounId()) {
  DCHECK(immutable_converter_);
  predictor_ = predictor_factory(*modules_, *this, *immutable_converter_);
  rewriter_ = rewriter_factory(*modules_);
  DCHECK(predictor_);
  DCHECK(rewriter_);
}

bool Converter::StartConversion(const ConversionRequest& request,
                                Segments* segments) const {
  DCHECK_EQ(request.request_type(), ConversionRequest::CONVERSION);

  absl::string_view key = request.key();
  if (key.empty()) {
    return false;
  }

  segments->InitForConvert(key);
  ApplyConversion(segments, request);

#ifdef _WIN32
  // AI リランキングを適用
  std::string history = GetHistoryContext(*segments);
  QueryAiConversionBatch(history, segments);
#endif

  return IsValidSegments(request, *segments);
}

bool Converter::StartReverseConversion(Segments* segments,
                                       const absl::string_view key) const {
  segments->Clear();
  if (key.empty()) {
    return false;
  }
  segments->InitForConvert(key);

  return reverse_converter_.ReverseConvert(key, segments);
}

// static
void Converter::MaybeSetConsumedKeySizeToCandidate(size_t consumed_key_size,
                                                   Candidate* candidate) {
  if (candidate->attributes & Attribute::PARTIALLY_KEY_CONSUMED) {
    return;
  }
  candidate->attributes |= Attribute::PARTIALLY_KEY_CONSUMED;
  candidate->consumed_key_size = consumed_key_size;
}

// static
void Converter::MaybeSetConsumedKeySizeToSegment(size_t consumed_key_size,
                                                 Segment* segment) {
  for (size_t i = 0; i < segment->candidates_size(); ++i) {
    MaybeSetConsumedKeySizeToCandidate(consumed_key_size,
                                       segment->mutable_candidate(i));
  }
  for (size_t i = 0; i < segment->meta_candidates_size(); ++i) {
    MaybeSetConsumedKeySizeToCandidate(consumed_key_size,
                                       segment->mutable_meta_candidate(i));
  }
}

namespace {
bool ValidateConversionRequestForPrediction(const ConversionRequest& request) {
  switch (request.request_type()) {
    case ConversionRequest::CONVERSION:
      return false;
    case ConversionRequest::PREDICTION:
    case ConversionRequest::SUGGESTION:
      return true;
    case ConversionRequest::PARTIAL_PREDICTION:
    case ConversionRequest::PARTIAL_SUGGESTION: {
      const size_t cursor = request.composer().GetCursor();
      return cursor != 0 || cursor != request.composer().GetLength();
    }
    default:
      ABSL_UNREACHABLE();
  }
}
}  // namespace

bool Converter::StartPrediction(const ConversionRequest& request,
                                Segments* segments) const {
  DCHECK(ValidateConversionRequestForPrediction(request));

  absl::string_view key = request.key();
  if (ShouldInitSegmentsForPrediction(key, *segments)) {
    segments->InitForConvert(key);
  }
  DCHECK_EQ(segments->conversion_segments_size(), 1);
  DCHECK_EQ(segments->conversion_segment(0).key(), key);

  if (!PredictForRequestWithSegments(request, segments)) {
    MOZC_VLOG(1) << "PredictForRequest failed for key: "
                 << segments->segment(0).key();
  }

#ifdef _WIN32
  // TAB補完（PREDICTION/SUGGESTIONリクエスト時）
  if (request.request_type() == ConversionRequest::PREDICTION ||
      request.request_type() == ConversionRequest::SUGGESTION) {
    std::string history = GetHistoryContext(*segments);
    std::string raw_input(key.data(), key.size());
    if (segments->conversion_segments_size() > 0) {
      QueryAiTabCompletion(history, raw_input,
                           segments->mutable_conversion_segment(0));
    }
  }
#endif

  ApplyPostProcessing(request, segments);
  return IsValidSegments(request, *segments);
}

bool Converter::StartPredictionWithPreviousSuggestion(
    const ConversionRequest& request, const Segment& previous_segment,
    Segments* segments) const {
  bool result = StartPrediction(request, segments);
  segments->PrependCandidates(previous_segment);
  if (!result) {
    return false;
  }

  ApplyPostProcessing(request, segments);
  return IsValidSegments(request, *segments);
}

void Converter::PrependCandidates(const ConversionRequest& request,
                                  const Segment& segment,
                                  Segments* segments) const {
  segments->PrependCandidates(segment);
  ApplyPostProcessing(request, segments);
}

void Converter::ApplyPostProcessing(const ConversionRequest& request,
                                    Segments* segments) const {
  RewriteAndSuppressCandidates(request, segments);
  TrimCandidates(request, segments);
  if (request.request_type() == ConversionRequest::PARTIAL_SUGGESTION ||
      request.request_type() == ConversionRequest::PARTIAL_PREDICTION) {
    MaybeSetConsumedKeySizeToSegment(Util::CharsLen(request.key()),
                                     segments->mutable_conversion_segment(0));
  }
}

// v7.0: FinishConversionに学習通知を追加
void Converter::FinishConversion(const ConversionRequest& request,
                                 Segments* segments) const {
#ifdef _WIN32
  // 確定情報をAIに通知（学習用）
  NotifyLearnToAI(*segments);
#endif

  for (Segment& segment : *segments) {
    if (segment.segment_type() == Segment::SUBMITTED) {
      segment.set_segment_type(Segment::FIXED_VALUE);
    }
    if (segment.candidates_size() > 0) {
      CompletePosIds(segment.mutable_candidate(0));
    }
  }

  PopulateReadingOfCommittedCandidateIfMissing(segments);

  absl::BitGen bitgen;
  const uint64_t revert_id = absl::Uniform<uint64_t>(
      absl::IntervalClosed, bitgen, 1, std::numeric_limits<uint64_t>::max());
  segments->set_revert_id(revert_id);

  const prediction::Result history_result = MakeHistoryResult(*segments);
  const std::vector<prediction::Result> committed_results =
      MakeLearningResults(*segments);
  const ConversionRequest finish_req = ConversionRequestBuilder()
                                           .SetConversionRequestView(request)
                                           .SetHistoryResultView(history_result)
                                           .Build();
  rewriter_->Finish(finish_req, *segments);
  predictor_->Finish(finish_req, committed_results, segments->revert_id());

  if (request.request_type() != ConversionRequest::CONVERSION &&
      segments->conversion_segments_size() >= 1 &&
      segments->conversion_segment(0).candidates_size() >= 1) {
    Segment* segment = segments->mutable_conversion_segment(0);
    segment->set_key(segment->candidate(0).key);
  }

  const int start_index = std::max<int>(
      0, segments->segments_size() - segments->max_history_segments_size());
  for (int i = 0; i < start_index; ++i) {
    segments->pop_front_segment();
  }

  for (Segment& segment : *segments) {
    segment.set_segment_type(Segment::HISTORY);
  }
}

void Converter::CancelConversion(Segments* segments) const {
  segments->clear_conversion_segments();
#ifdef _WIN32
  g_learn_tracker.Clear();
#endif
}

void Converter::ResetConversion(Segments* segments) const { 
  segments->Clear();
#ifdef _WIN32
  g_learn_tracker.Clear();
#endif
}

void Converter::RevertConversion(Segments* segments) const {
  if (segments->revert_id() == 0) {
    return;
  }
  rewriter_->Revert(*segments);
  predictor_->Revert(segments->revert_id());
  segments->set_revert_id(0);
}

bool Converter::DeleteCandidateFromHistory(const Segments& segments,
                                           size_t segment_index,
                                           int candidate_index) const {
  DCHECK_LT(segment_index, segments.segments_size());
  const Segment& segment = segments.segment(segment_index);
  DCHECK(segment.is_valid_index(candidate_index));
  const Candidate& candidate = segment.candidate(candidate_index);
  bool result = false;
  result |=
      rewriter_->ClearHistoryEntry(segments, segment_index, candidate_index);
  result |= predictor_->ClearHistoryEntry(candidate.key, candidate.value);

  return result;
}

bool Converter::ReconstructHistory(
    Segments* segments, const absl::string_view preceding_text) const {
  segments->Clear();
  return history_reconstructor_.ReconstructHistory(preceding_text, segments);
}

bool Converter::CommitSegmentValueInternal(
    Segments* segments, size_t segment_index, int candidate_index,
    Segment::SegmentType segment_type) const {
  segment_index = GetSegmentIndex(segments, segment_index);
  if (segment_index == kErrorIndex) {
    return false;
  }

  Segment* segment = segments->mutable_segment(segment_index);
  const int values_size = static_cast<int>(segment->candidates_size());
  if (candidate_index < -transliteration::NUM_T13N_TYPES ||
      candidate_index >= values_size) {
    return false;
  }

  segment->set_segment_type(segment_type);
  segment->move_candidate(candidate_index, 0);

  if (candidate_index != 0) {
    segment->mutable_candidate(0)->attributes |= Attribute::RERANKED;
  }

  return true;
}

bool Converter::CommitSegmentValue(Segments* segments, size_t segment_index,
                                   int candidate_index) const {
  return CommitSegmentValueInternal(segments, segment_index, candidate_index,
                                    Segment::FIXED_VALUE);
}

bool Converter::CommitPartialSuggestionSegmentValue(
    Segments* segments, size_t segment_index, int candidate_index,
    const absl::string_view current_segment_key,
    const absl::string_view new_segment_key) const {
  DCHECK_GT(segments->conversion_segments_size(), 0);

  const size_t raw_segment_index = GetSegmentIndex(segments, segment_index);
  if (!CommitSegmentValueInternal(segments, segment_index, candidate_index,
                                  Segment::SUBMITTED)) {
    return false;
  }

  Segment* segment = segments->mutable_segment(raw_segment_index);
  DCHECK_LT(0, segment->candidates_size());
  segment->set_key(current_segment_key);

  Segment* new_segment = segments->insert_segment(raw_segment_index + 1);
  new_segment->set_key(new_segment_key);
  DCHECK_GT(segments->conversion_segments_size(), 0);

  return true;
}

bool Converter::FocusSegmentValue(Segments* segments, size_t segment_index,
                                  int candidate_index) const {
  segment_index = GetSegmentIndex(segments, segment_index);
  if (segment_index == kErrorIndex) {
    return false;
  }

  return rewriter_->Focus(segments, segment_index, candidate_index);
}

bool Converter::CommitSegments(Segments* segments,
                               absl::Span<const size_t> candidate_index) const {
  for (size_t i = 0; i < candidate_index.size(); ++i) {
    if (!CommitSegmentValueInternal(segments, 0, candidate_index[i],
                                    Segment::SUBMITTED)) {
      return false;
    }
  }
  return true;
}

bool Converter::ResizeSegment(Segments* segments,
                              const ConversionRequest& request,
                              size_t segment_index, int offset_length) const {
  if (request.request_type() != ConversionRequest::CONVERSION) {
    return false;
  }

  if (offset_length == 0) {
    return false;
  }

  if (segment_index >= segments->conversion_segments_size()) {
    return false;
  }

  const size_t key_len = segments->conversion_segment(segment_index).key_len();
  if (key_len == 0) {
    return false;
  }

  const int new_size = key_len + offset_length;
  if (new_size <= 0 || new_size > std::numeric_limits<uint8_t>::max()) {
    return false;
  }
  const std::array<uint8_t, 1> new_size_array = {
      static_cast<uint8_t>(new_size)};
  return ResizeSegments(segments, request, segment_index, new_size_array);
}

bool Converter::ResizeSegments(Segments* segments,
                               const ConversionRequest& request,
                               size_t start_segment_index,
                               absl::Span<const uint8_t> new_size_array) const {
  if (request.request_type() != ConversionRequest::CONVERSION) {
    return false;
  }

  start_segment_index = GetSegmentIndex(segments, start_segment_index);
  if (start_segment_index == kErrorIndex) {
    return false;
  }

  if (!segments->Resize(start_segment_index, new_size_array)) {
    return false;
  }

  ApplyConversion(segments, request);
  return true;
}

void Converter::ApplyConversion(Segments* segments,
                                const ConversionRequest& request) const {
  if (!immutable_converter_->Convert(request, segments)) {
    MOZC_VLOG(1) << "Convert failed for key: " << segments->segment(0).key();
  }

  ApplyPostProcessing(request, segments);
}

void Converter::CompletePosIds(Candidate* candidate) const {
  DCHECK(candidate);
  if (candidate->value.empty() || candidate->key.empty()) {
    return;
  }

  if (candidate->lid != 0 && candidate->rid != 0) {
    return;
  }

  candidate->lid = general_noun_id_;
  candidate->rid = general_noun_id_;
  constexpr size_t kExpandSizeStart = 5;
  constexpr size_t kExpandSizeDiff = 50;
  constexpr size_t kExpandSizeMax = 80;
  for (size_t size = kExpandSizeStart; size < kExpandSizeMax;
       size += kExpandSizeDiff) {
    Segments segments;
    segments.InitForConvert(candidate->key);
    const ConversionRequest request =
        ConversionRequestBuilder()
            .SetOptions({
                .request_type = ConversionRequest::PREDICTION,
                .max_conversion_candidates_size = static_cast<int>(size),
            })
            .Build();
    if (!immutable_converter_->Convert(request, &segments)) {
      LOG(ERROR) << "ImmutableConverter::Convert() failed";
      return;
    }
    for (size_t i = 0; i < segments.segment(0).candidates_size(); ++i) {
      const Candidate& ref_candidate = segments.segment(0).candidate(i);
      if (ref_candidate.value == candidate->value) {
        candidate->lid = ref_candidate.lid;
        candidate->rid = ref_candidate.rid;
        candidate->cost = ref_candidate.cost;
        candidate->wcost = ref_candidate.wcost;
        candidate->structure_cost = ref_candidate.structure_cost;
        MOZC_VLOG(1) << "Set LID: " << candidate->lid;
        MOZC_VLOG(1) << "Set RID: " << candidate->rid;
        return;
      }
    }
  }
  MOZC_DVLOG(2) << "Cannot set lid/rid. use default value. "
                << "key: " << candidate->key << ", "
                << "value: " << candidate->value << ", "
                << "lid: " << candidate->lid << ", "
                << "rid: " << candidate->rid;
}

void Converter::RewriteAndSuppressCandidates(const ConversionRequest& request,
                                             Segments* segments) const {
  // 1. Resize segments if needed.
  if (std::optional<RewriterInterface::ResizeSegmentsRequest> resize_request =
          rewriter_->CheckResizeSegmentsRequest(request, *segments);
      resize_request.has_value()) {
    if (ResizeSegments(segments, request, resize_request->segment_index,
                       resize_request->segment_sizes)) {
      return;
    }
  }

  // 2. Rewrite candidates in each segment.
  if (!rewriter_->Rewrite(request, segments)) {
    return;
  }

  // 3. Suppress candidates in each segment.
  if (!user_dictionary_.HasSuppressedEntries()) {
    return;
  }
  for (Segment& segment : segments->conversion_segments()) {
    for (size_t j = 0; j < segment.candidates_size();) {
      const Candidate& cand = segment.candidate(j);
      if (user_dictionary_.IsSuppressedEntry(cand.key, cand.value)) {
        segment.erase_candidate(j);
      } else {
        ++j;
      }
    }
  }
}

void Converter::TrimCandidates(const ConversionRequest& request,
                               Segments* segments) const {
  const mozc::commands::Request& request_proto = request.request();
  if (!request_proto.has_candidates_size_limit()) {
    return;
  }

  const int limit = request_proto.candidates_size_limit();
  for (Segment& segment : segments->conversion_segments()) {
    const int candidates_size = segment.candidates_size();
    const int candidates_limit =
        std::max<int>(1, limit - segment.meta_candidates_size());
    if (candidates_size < candidates_limit) {
      continue;
    }
    segment.erase_candidates(candidates_limit,
                             candidates_size - candidates_limit);
  }
}

bool Converter::Reload() {
  modules().GetUserDictionary().Reload();
  return rewriter().Reload() && predictor().Reload();
}

bool Converter::Sync() { return rewriter().Sync() && predictor().Sync(); }

bool Converter::Wait() {
  modules().GetUserDictionary().WaitForReloader();
  return predictor().Wait();
}

std::optional<std::string> Converter::GetReading(absl::string_view text) const {
  Segments segments;
  if (!StartReverseConversion(&segments, text)) {
    LOG(ERROR) << "Reverse conversion failed to get the reading of " << text;
    return std::nullopt;
  }
  if (segments.conversion_segments_size() != 1 ||
      segments.conversion_segment(0).candidates_size() == 0) {
    LOG(ERROR) << "Reverse conversion returned an invalid result for " << text;
    return std::nullopt;
  }
  return std::move(
      segments.mutable_conversion_segment(0)->mutable_candidate(0)->value);
}

void Converter::PopulateReadingOfCommittedCandidateIfMissing(
    Segments* segments) const {
  if (segments->conversion_segments_size() == 0) return;

  Segment* segment = segments->mutable_conversion_segment(0);
  if (segment->candidates_size() == 0) return;

  Candidate* cand = segment->mutable_candidate(0);
  if (!cand->key.empty() || cand->value.empty()) return;

  if (cand->content_value == cand->value) {
    if (std::optional<std::string> key = GetReading(cand->value);
        key.has_value()) {
      cand->key = *key;
      cand->content_key = *std::move(key);
    }
    return;
  }

  if (cand->content_value.empty()) {
    LOG(ERROR) << "Content value is empty: " << *cand;
    return;
  }

  const absl::string_view functional_value = cand->functional_value();
  if (Util::GetScriptType(functional_value) != Util::HIRAGANA) {
    LOG(ERROR) << "The functional value is not hiragana: " << *cand;
    return;
  }
  if (std::optional<std::string> content_key = GetReading(cand->content_value);
      content_key.has_value()) {
    cand->key = absl::StrCat(*content_key, functional_value);
    cand->content_key = *std::move(content_key);
  }
}

bool Converter::PredictForRequestWithSegments(const ConversionRequest& request,
                                              Segments* segments) const {
  DCHECK(segments);
  DCHECK(predictor_);

  const prediction::Result history_result = MakeHistoryResult(*segments);

  const ConversionRequest conv_req = ConversionRequestBuilder()
                                         .SetConversionRequestView(request)
                                         .SetHistoryResultView(history_result)
                                         .Build();

  const std::vector<prediction::Result> results = predictor_->Predict(conv_req);

  Segment* segment = segments->mutable_conversion_segment(0);
  DCHECK(segment);

  for (const prediction::Result& result : results) {
    Candidate* candidate = segment->add_candidate();
    strings::Assign(candidate->key, result.key);
    strings::Assign(candidate->value, result.value);
    strings::Assign(candidate->description, result.description);
    strings::Assign(candidate->display_value, result.display_value);
    candidate->lid = result.lid;
    candidate->rid = result.rid;
    candidate->wcost = result.wcost;
    candidate->cost = result.cost;
    candidate->attributes = result.candidate_attributes;
    candidate->consumed_key_size = result.consumed_key_size;
    candidate->inner_segment_boundary = result.inner_segment_boundary;

    std::tie(candidate->content_key, candidate->content_value) =
        result.inner_segments().GetMergedContentKeyAndValue();
#ifndef NDEBUG
    absl::StrAppend(&candidate->log, "\n", result.log);
#endif  // NDEBUG
  }

  return !results.empty();
}

// static
std::vector<prediction::Result> Converter::MakeLearningResults(
    const Segments& segments) {
  std::vector<prediction::Result> results;

  if (segments.conversion_segments_size() == 0) {
    return results;
  }

  if (segments.conversion_segments_size() == 1) {
    constexpr int kMaxHistorySize = 5;
    for (const auto& candidate : segments.conversion_segment(0).candidates()) {
      prediction::Result result;
      strings::Assign(result.key, candidate->key);
      strings::Assign(result.value, candidate->value);
      strings::Assign(result.description, candidate->description);
      strings::Assign(result.display_value, candidate->display_value);
      result.lid = candidate->lid;
      result.rid = candidate->rid;
      result.wcost = candidate->wcost;
      result.cost = candidate->cost;
      result.candidate_attributes = candidate->attributes;
      result.consumed_key_size = candidate->consumed_key_size;
      result.inner_segment_boundary = candidate->inner_segment_boundary;
      if (result.inner_segment_boundary.empty()) {
        result.inner_segment_boundary = BuildInnerSegmentBoundary(
            {{candidate->key.size(), candidate->value.size(),
              candidate->content_key.size(), candidate->content_value.size()}},
            result.key, result.value);
      }
      results.emplace_back(std::move(result));
      if (results.size() >= kMaxHistorySize) break;
    }

    return results;
  }

  {
    prediction::Result result;
    InnerSegmentBoundaryBuilder builder;
    for (const auto& segment : segments.conversion_segments()) {
      if (segment.candidates_size() == 0) return {};
      const Candidate& candidate = segment.candidate(0);
      absl::StrAppend(&result.key, candidate.key);
      absl::StrAppend(&result.value, candidate.value);
      result.candidate_attributes |= candidate.attributes;
      result.wcost += candidate.wcost;
      result.cost += candidate.cost;
      builder.Add(candidate.key.size(), candidate.value.size(),
                  candidate.content_key.size(), candidate.content_value.size());
    }
    result.inner_segment_boundary = builder.Build(result.key, result.value);
    result.lid = segments.conversion_segments().front().candidate(0).lid;
    result.rid = segments.conversion_segments().back().candidate(0).rid;

    results.emplace_back(std::move(result));
  }

  return results;
}

// static
prediction::Result Converter::MakeHistoryResult(const Segments& segments) {
  prediction::Result result;

  if (segments.history_segments_size() == 0) {
    return result;
  }

  InnerSegmentBoundaryBuilder builder;
  for (const auto& segment : segments.history_segments()) {
    if (segment.candidates_size() == 0) {
      return prediction::Result::DefaultResult();
    }
    const Candidate& candidate = segment.candidate(0);
    absl::StrAppend(&result.key, candidate.key);
    absl::StrAppend(&result.value, candidate.value);
    result.candidate_attributes |= candidate.attributes;
    builder.Add(candidate.key.size(), candidate.value.size(),
                candidate.content_key.size(), candidate.content_value.size());
  }

  result.inner_segment_boundary = builder.Build(result.key, result.value);

  const int size = segments.history_segments_size();
  result.lid = segments.history_segment(0).candidate(0).lid;
  result.rid = segments.history_segment(size - 1).candidate(0).rid;
  result.cost = segments.history_segment(size - 1).candidate(0).cost;

  return result;
}

}  // namespace converter
}  // namespace mozc
