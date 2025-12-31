// Copyright 2010-2021, Google Inc.
// All rights reserved.
//
// Mozc AI IME v5.0 - Enhanced AI Integration
// 
// 変更点 v5.0:
// - FinishConversionに学習通知を追加（すべての確定フローに対応）
// - NotifyLearnToAIFromSegments関数を追加
// - CommitSegmentsの学習通知も維持（部分コミット用）

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
#include <sstream>

#ifdef _WIN32
#include <windows.h>
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
// AI Server Client v5.0
// ==========================================

#ifdef _WIN32

inline void DebugLog(const char* format, ...) {
    char buffer[2048];
    va_list args;
    va_start(args, format);
    vsnprintf(buffer, sizeof(buffer), format, args);
    va_end(args);
    ::OutputDebugStringA(buffer);
}

// パイプ名定義
static const char* const kAiPipeName = "\\\\.\\pipe\\MozcBertPipe";
static const char* const kAiLearnPipeName = "\\\\.\\pipe\\MozcAILearnPipe";
static const char* const kAiCorrectPipeName = "\\\\.\\pipe\\MozcAICorrectPipe";  // 誤字補正用

// タイムアウト設定
static const DWORD kPipeTimeoutMs = 100;
static const DWORD kCorrectPipeTimeoutMs = 200;  // 補正用は少し長め
static const DWORD kLearnPipeTimeoutMs = 50;     // 学習用は短め

// 候補数制限
static const int kMaxCandidatesPerSegment = 4;

/**
 * カタカナかどうか判定
 */
bool IsKatakana(const std::string& text) {
    // UTF-8でカタカナ範囲をチェック
    for (size_t i = 0; i < text.size(); ) {
        unsigned char c = text[i];
        if ((c & 0x80) == 0) {
            // ASCII
            return false;
        } else if ((c & 0xE0) == 0xC0) {
            i += 2;
        } else if ((c & 0xF0) == 0xE0) {
            // 3バイト文字 - カタカナ範囲チェック
            if (i + 2 < text.size()) {
                unsigned char c1 = text[i];
                unsigned char c2 = text[i + 1];
                unsigned char c3 = text[i + 2];
                // カタカナ: U+30A0-U+30FF (E3 82 A0 - E3 83 BF)
                // 半角カタカナ: U+FF65-U+FF9F (EF BD A5 - EF BE 9F)
                bool is_katakana = (c1 == 0xE3 && c2 >= 0x82 && c2 <= 0x83);
                bool is_hw_katakana = (c1 == 0xEF && ((c2 == 0xBD && c3 >= 0xA5) || (c2 == 0xBE && c3 <= 0x9F)));
                if (!is_katakana && !is_hw_katakana) {
                    return false;
                }
            }
            i += 3;
        } else if ((c & 0xF8) == 0xF0) {
            i += 4;
        } else {
            i++;
        }
    }
    return !text.empty();
}

/**
 * 英語モードフラグを判定
 * - カタカナのみの入力
 * - 候補にカタカナ語が多い
 */
int DetectEnglishModeFlag(const Segment& seg) {
    std::string reading(seg.key().data(), seg.key().size());
    
    // 読みがカタカナのみ
    if (IsKatakana(reading)) {
        return 1;
    }
    
    // 候補にカタカナが2つ以上
    int katakana_count = 0;
    for (size_t j = 0; j < std::min<size_t>(seg.candidates_size(), 5); ++j) {
        if (IsKatakana(seg.candidate(j).value)) {
            katakana_count++;
        }
    }
    if (katakana_count >= 2) {
        return 1;
    }
    
    return 0;
}

/**
 * AIサーバーに全文節を一括送信 v4
 * 
 * 送信フォーマット:
 *   {history}\t{seg_count}\t{reading1}\t{eng_flag1}\t{cand_count1}\t{cand1}...\t{reading2}...
 * 
 * 受信フォーマット:
 *   {selected1}\t{selected2}\t...
 */
bool QueryAiConversionBatch(const std::string& history_context, 
                            Segments* segments) {
    if (segments->conversion_segments_size() == 0) {
        return false;
    }
    
    size_t seg_count = segments->conversion_segments_size();
    
    std::ostringstream payload;
    payload << history_context << "\t";
    payload << seg_count;
    
    std::ostringstream debug_send;
    debug_send << "[MozcAI] >>> hist:" << history_context.length() << " segs:" << seg_count << " [";
    
    for (size_t i = 0; i < seg_count; ++i) {
        const Segment& seg = segments->conversion_segment(i);
        
        std::string reading(seg.key().data(), seg.key().size());
        int eng_flag = DetectEnglishModeFlag(seg);
        
        payload << "\t" << reading;
        payload << "\t" << eng_flag;  // 英語モードフラグ追加
        
        int cand_count = std::min(static_cast<int>(seg.candidates_size()), 
                                   kMaxCandidatesPerSegment);
        payload << "\t" << cand_count;
        
        debug_send << reading;
        if (eng_flag) debug_send << "[E]";
        debug_send << "(";
        
        for (int j = 0; j < cand_count; ++j) {
            std::string cand_val = seg.candidate(j).value;
            payload << "\t" << cand_val;
            if (j < 3) {
                debug_send << cand_val;
                if (j < 2 && j < cand_count - 1) debug_send << "/";
            }
        }
        debug_send << ")";
        if (i < seg_count - 1) debug_send << " ";
    }
    debug_send << "]\n";
    DebugLog("%s", debug_send.str().c_str());
    
    std::string payload_str = payload.str();
    
    char buffer[65536];
    DWORD bytes_read = 0;
    
    BOOL success = ::CallNamedPipeA(
        kAiPipeName,
        const_cast<char*>(payload_str.c_str()),
        static_cast<DWORD>(payload_str.size()),
        buffer,
        sizeof(buffer),
        &bytes_read,
        kPipeTimeoutMs
    );
    
    if (!success || bytes_read == 0) {
        DebugLog("[MozcAI] Pipe failed: err=%d\n", ::GetLastError());
        return false;
    }
    
    std::string response(buffer, bytes_read);
    std::vector<std::string> selected_values;
    
    {
        std::istringstream iss(response);
        std::string token;
        while (std::getline(iss, token, '\t')) {
            selected_values.push_back(token);
        }
    }
    
    std::ostringstream debug_recv;
    debug_recv << "[MozcAI] <<< ";
    for (size_t i = 0; i < selected_values.size(); ++i) {
        debug_recv << "[" << selected_values[i] << "]";
    }
    debug_recv << "\n";
    DebugLog("%s", debug_recv.str().c_str());
    
    size_t applied_count = 0;
    size_t same_count = 0;
    
    for (size_t i = 0; i < seg_count && i < selected_values.size(); ++i) {
        const std::string& selected = selected_values[i];
        if (selected.empty()) {
            continue;
        }
        
        Segment* seg = segments->mutable_conversion_segment(i);
        
        int found_index = -1;
        for (size_t j = 0; j < seg->candidates_size(); ++j) {
            if (seg->candidate(j).value == selected) {
                found_index = static_cast<int>(j);
                break;
            }
        }
        
        if (found_index > 0) {
            seg->move_candidate(found_index, 0);
            seg->mutable_candidate(0)->attributes |= Attribute::RERANKED;
            applied_count++;
        } else if (found_index == 0) {
            same_count++;
        } else {
            // AI生成の新候補を追加
            std::string reading(seg->key().data(), seg->key().size());
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
            
            if (seg->candidates_size() > 1) {
                seg->move_candidate(seg->candidates_size() - 1, 0);
            }
            applied_count++;
        }
    }
    
    DebugLog("[MozcAI] Result: applied=%zu same=%zu total=%zu\n", 
             applied_count, same_count, seg_count);
    
    return applied_count > 0 || same_count > 0;
}

/**
 * 履歴コンテキストを取得
 */
std::string GetHistoryContext(const Segments& segments) {
    std::string history;
    for (size_t i = 0; i < segments.history_segments_size(); ++i) {
        const Segment& seg = segments.history_segment(i);
        if (seg.candidates_size() > 0) {
            history += seg.candidate(0).value;
        }
    }
    return history;
}

/**
 * 確定情報をAIサーバーに通知（学習用） - CommitSegments用
 * candidate_indicesを使用してどの候補が確定されたかを判断
 */
void NotifyLearnToAI(const Segments& segments, 
                     absl::Span<const size_t> candidate_indices) {
    if (candidate_indices.empty()) {
        return;
    }
    
    size_t conv_seg_count = segments.conversion_segments_size();
    if (conv_seg_count == 0) {
        return;
    }
    
    std::ostringstream payload;
    payload << "LEARN";
    
    size_t sent_count = 0;
    for (size_t i = 0; i < candidate_indices.size() && i < conv_seg_count; ++i) {
        const Segment& seg = segments.conversion_segment(i);
        size_t cand_idx = candidate_indices[i];
        
        if (cand_idx < seg.candidates_size()) {
            std::string reading(seg.key().data(), seg.key().size());
            std::string value = seg.candidate(cand_idx).value;
            
            payload << "\t" << reading << "\t" << value;
            sent_count++;
        }
    }
    
    if (sent_count == 0) {
        return;
    }
    
    std::string payload_str = payload.str();
    
    DebugLog("[MozcAI] Learn (CommitSegments): %s\n", payload_str.c_str());
    
    char buffer[256];
    DWORD bytes_read = 0;
    
    BOOL result = ::CallNamedPipeA(
        kAiLearnPipeName,
        const_cast<char*>(payload_str.c_str()),
        static_cast<DWORD>(payload_str.size()),
        buffer,
        sizeof(buffer),
        &bytes_read,
        kLearnPipeTimeoutMs
    );
    
    if (result) {
        DebugLog("[MozcAI] Learn notify (CommitSegments): %zu segments\n", sent_count);
    }
}

/**
 * 確定情報をAIサーバーに通知（学習用） - FinishConversion用
 * セグメントの最初の候補（確定されたもの）を使用
 * 
 * v5.0新規追加:
 * この関数はFinishConversionから呼ばれ、すべての確定フローに対応します。
 * - Session::Commit() → EngineConverter::Commit() → FinishConversion()
 * - Session::CommitSuggestion() → EngineConverter::CommitSuggestionInternal() → FinishConversion()
 */
void NotifyLearnToAIFromSegments(const Segments& segments) {
    size_t conv_seg_count = segments.conversion_segments_size();
    if (conv_seg_count == 0) {
        return;
    }
    
    std::ostringstream payload;
    payload << "LEARN";
    
    std::ostringstream debug_msg;
    debug_msg << "[MozcAI] Learn (Finish): ";
    
    size_t sent_count = 0;
    for (size_t i = 0; i < conv_seg_count; ++i) {
        const Segment& seg = segments.conversion_segment(i);
        
        if (seg.candidates_size() > 0) {
            std::string reading(seg.key().data(), seg.key().size());
            // 最初の候補が確定されたもの
            std::string value = seg.candidate(0).value;
            
            payload << "\t" << reading << "\t" << value;
            debug_msg << "[" << reading << "->" << value << "]";
            sent_count++;
        }
    }
    
    if (sent_count == 0) {
        return;
    }
    
    std::string payload_str = payload.str();
    
    DebugLog("%s\n", debug_msg.str().c_str());
    
    char buffer[256];
    DWORD bytes_read = 0;
    
    BOOL result = ::CallNamedPipeA(
        kAiLearnPipeName,
        const_cast<char*>(payload_str.c_str()),
        static_cast<DWORD>(payload_str.size()),
        buffer,
        sizeof(buffer),
        &bytes_read,
        kLearnPipeTimeoutMs
    );
    
    if (result) {
        DebugLog("[MozcAI] Learn notify (FinishConversion): %zu segments sent\n", sent_count);
    } else {
        DebugLog("[MozcAI] Learn pipe failed (FinishConversion): err=%d\n", ::GetLastError());
    }
}

/**
 * 誤字補正リクエスト（TAB補完用）
 * 
 * 送信: CORRECT\t{確定済みテキスト}\t{文字数}
 * 受信: {補正候補1}\t{補正候補2}... または NONE
 */
std::vector<std::string> QueryAiCorrection(const std::string& committed_text, 
                                            int check_chars = 50) {
    std::vector<std::string> corrections;
    
    if (committed_text.length() < 5) {
        return corrections;
    }
    
    std::ostringstream payload;
    payload << "CORRECT\t" << committed_text << "\t" << check_chars;
    
    std::string payload_str = payload.str();
    
    char buffer[4096];
    DWORD bytes_read = 0;
    
    BOOL success = ::CallNamedPipeA(
        kAiCorrectPipeName,
        const_cast<char*>(payload_str.c_str()),
        static_cast<DWORD>(payload_str.size()),
        buffer,
        sizeof(buffer),
        &bytes_read,
        kCorrectPipeTimeoutMs
    );
    
    if (!success || bytes_read == 0) {
        return corrections;
    }
    
    std::string response(buffer, bytes_read);
    if (response == "NONE" || response.empty()) {
        return corrections;
    }
    
    std::istringstream iss(response);
    std::string token;
    while (std::getline(iss, token, '\t')) {
        if (!token.empty()) {
            corrections.push_back(token);
        }
    }
    
    return corrections;
}

#else

// Non-Windows stubs
inline void DebugLog(const char* format, ...) {}
bool QueryAiConversionBatch(const std::string& history_context, Segments* segments) { return false; }
std::string GetHistoryContext(const Segments& segments) { return ""; }
void NotifyLearnToAI(const Segments& segments, absl::Span<const size_t> candidate_indices) {}
void NotifyLearnToAIFromSegments(const Segments& segments) {}
std::vector<std::string> QueryAiCorrection(const std::string& committed_text, int check_chars = 50) { 
    return std::vector<std::string>(); 
}

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

void SetKey(Segments* segments, const absl::string_view key) {
  segments->set_max_history_segments_size(4);
  segments->clear_conversion_segments();
  segments->set_resized(false);
  Segment* seg = segments->add_segment();
  DCHECK(seg);
  seg->set_key(key);
  seg->set_segment_type(Segment::FREE);
}

bool ShouldSetKeyForPrediction(const ConversionRequest& request,
                               const Segments& segments) {
  // If the segments is for conversion or already has some results,
  // the new key should not be set to keep the current state.
  // NOTE:
  // 1. Not resized
  //    Following the first condition, segments may be CONVERSION_SEGMENTS_SIZE
  //    > 1 if the last segment is very long and the Converter tries to resplit
  //    it. In this case, the old key should be kept, because it's not
  //    triggered by a user, rather it's triggered by Predictor.
  // 2. Not submitted
  //    After committing a segment, the converter generates next segments
  //    for the rest of the composition. We should keep them until a new
  //    key is set. Note if the next segment is generated, segments has some
  //    conversion result.
  return !segments.resized() &&
         (segments.conversion_segments_size() == 0 ||
          segments.conversion_segment(0).candidates_size() == 0);
}

bool IsValidSegments(const ConversionRequest& request,
                     const Segments& segments) {
  // All segments should have at least one candidate.  Otherwise prediction
  // results may get dropped.
  for (const Segment& segment : segments.conversion_segments()) {
    if (segment.candidates_size() == 0) {
      return false;
    }
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
      predictor_(predictor_factory(*modules_, *this, *immutable_converter_)),
      rewriter_(rewriter_factory(*modules_)),
      pos_matcher_(*modules_->GetPosMatcher()),
      user_dictionary_(*modules_->GetUserDictionary()),
      history_reconstructor_(*modules_->GetPosMatcher()),
      reverse_converter_(*modules_->GetDictionary(), *modules_->GetPosMatcher()),
      general_noun_id_(modules_->GetPosMatcher()->GetGeneralNounId()) {}

bool Converter::StartConversion(const ConversionRequest& request,
                                Segments* segments) const {
  SetKey(segments, request.key());
  ApplyConversion(segments, request);
  
#ifdef _WIN32
  // Apply AI reranking
  std::string history = GetHistoryContext(*segments);
  QueryAiConversionBatch(history, segments);
#endif
  
  return IsValidSegments(request, *segments);
}

bool Converter::StartReverseConversion(Segments* segments,
                                       const absl::string_view key) const {
  return reverse_converter_.ReverseConvert(key, segments);
}

bool Converter::StartPrediction(const ConversionRequest& request,
                                Segments* segments) const {
  DCHECK(segments);
  if (ShouldSetKeyForPrediction(request, *segments)) {
    SetKey(segments, request.key());
  }
  return PredictForRequestWithSegments(request, segments);
}

bool Converter::StartPredictionWithPreviousSuggestion(
    const ConversionRequest& request, const Segment& previous_segment,
    Segments* segments) const {
  DCHECK(segments);
  if (ShouldSetKeyForPrediction(request, *segments)) {
    SetKey(segments, request.key());
  }
  PrependCandidates(request, previous_segment, segments);
  return PredictForRequestWithSegments(request, segments);
}

void Converter::PrependCandidates(const ConversionRequest& request,
                                  const Segment& previous_segment,
                                  Segments* segments) const {
  // This function just prepends all previous candidates in order.
  if (segments->conversion_segments_size() == 0) {
    Segment* segment = segments->add_segment();
    segment->set_segment_type(Segment::FREE);
    segment->set_key(request.key());
  }
  auto* segment = segments->mutable_conversion_segment(0);
  for (auto iter = previous_segment.candidates().rbegin();
       iter != previous_segment.candidates().rend(); ++iter) {
    // Move this candidate to the top.
    // Instead of directly prepending a new candidate by calling
    // segment->insert_candidate(0), first look for the duplicate candidate from
    // the existing candidates.  If found, move its index to the top.
    bool found = false;
    for (size_t i = 0; i < segment->candidates_size(); ++i) {
      if (iter->value == segment->candidate(i).value) {
        segment->move_candidate(i, 0);
        found = true;
        break;
      }
    }
    if (!found) {
      segment->insert_candidate(0, *iter);
    }
  }
}

// v5.0: FinishConversionに学習通知を追加
void Converter::FinishConversion(const ConversionRequest& request,
                                 Segments* segments) const {
#ifdef _WIN32
  // 確定情報をAIに通知（学習用）
  // この通知はすべての確定フローで呼ばれます：
  // - EngineConverter::Commit() → FinishConversion()
  // - EngineConverter::CommitSuggestionInternal() → FinishConversion()
  NotifyLearnToAIFromSegments(*segments);
#endif

  absl::BitGen gen;
  constexpr int kCandidateSize = 3;
  uint64_t revert_id = (static_cast<uint64_t>(absl::Uniform<uint32_t>(gen))
                        << 32) |
                       static_cast<uint64_t>(absl::Uniform<uint32_t>(gen));
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
}

void Converter::ResetConversion(Segments* segments) const { segments->Clear(); }

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

  if (candidate_index >= 0 && candidate_index < values_size) {
    segment->move_candidate(candidate_index, 0);
  } else if (candidate_index < 0) {
    segment->set_segment_type(segment_type);
    return true;
  }

  segment->set_segment_type(segment_type);
  MaybeSetConsumedKeySizeToSegment(Util::CharsLen(segment->candidate(0).key),
                                   segment);
  CompletePosIds(segment->mutable_candidate(0));
  return true;
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
  for (Candidate& candidate : *segment->mutable_candidates()) {
    MaybeSetConsumedKeySizeToCandidate(consumed_key_size, &candidate);
  }
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
#ifdef _WIN32
  // 確定情報をAIに通知（部分コミット用）
  // この関数はCommitSegmentsInternal経由で呼ばれる（CommitFirstSegment, CommitHeadToFocusedSegments）
  NotifyLearnToAI(*segments, candidate_index);
#endif

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

  // TODO(taku): adjust inner_segment_boundary
  segment_index = GetSegmentIndex(segments, segment_index);
  if (segment_index == kErrorIndex) {
    return false;
  }

  // Segment boundary correction ensures the following invariants for the
  // conversion segment [i, size) where i is the segment_index:
  //   - The sum of all segment keys is kept.
  //   - The correction can only update segments in [i, size).
  // Therefore, resize is simply applied to the segment [i] for simplicity
  // and let the immutable_converter decide how to split the remaining key.
  Segment* seg = segments->mutable_segment(segment_index);
  const absl::string_view old_key = seg->key();
  const int old_key_size = Util::CharsLen(old_key);
  const int new_key_size = old_key_size + offset_length;
  if (new_key_size <= 0) {
    return false;
  }
  const absl::string_view new_key = Util::Utf8SubString(old_key, 0, new_key_size);
  const bool is_shrink = (offset_length < 0);

  // Concatenate the remaining part (if any) and all the following segment keys.
  std::string remaining_key(
      Util::Utf8SubString(old_key, new_key_size, std::string::npos));
  for (size_t i = segment_index + 1; i < segments->segments_size(); ++i) {
    remaining_key.append(segments->segment(i).key());
  }
  // Pop back all the following conversion segments, i.e., [i+1, size].
  while (segments->segments_size() > segment_index + 1 &&
         segments->mutable_segment(segments->segments_size() - 1)->is_valid()) {
    segments->pop_back_segment();
  }

  seg->clear_candidates();
  seg->clear_meta_candidates();
  seg->set_key(new_key);
  seg->set_segment_type(Segment::FREE);

  // Update remaining_key as segment.
  if (!remaining_key.empty()) {
    Segment* new_seg = segments->add_segment();
    new_seg->set_key(remaining_key);
    new_seg->set_segment_type(Segment::FREE);
  }

  ApplyConversion(segments, request);

  if (is_shrink) {
    // Move the shrunken segment while fixing.
    seg = segments->mutable_segment(segment_index);
    for (Candidate& candidate : *seg->mutable_candidates()) {
      MaybeSetConsumedKeySizeToCandidate(Util::CharsLen(new_key), &candidate);
    }
  }

  segments->set_resized(true);

  return true;
}

bool Converter::ResizeSegments(Segments* segments,
                               const ConversionRequest& request,
                               size_t start_segment_index,
                               absl::Span<const uint8_t> new_size_array) const {
  if (request.request_type() != ConversionRequest::CONVERSION) {
    return false;
  }

  // TODO(taku): adjust inner_segment_boundary

  start_segment_index = GetSegmentIndex(segments, start_segment_index);
  if (start_segment_index == kErrorIndex) {
    return false;
  }

  // This is the simplified resizing: first concatenate all keys of segments in
  // [start_segment_index, segments_size), and then resplit the keys using
  // new_size_array.
  std::string concatenated_key;
  for (size_t i = start_segment_index; i < segments->segments_size(); ++i) {
    concatenated_key.append(segments->segment(i).key());
  }
  size_t sum = 0;
  for (const uint8_t n : new_size_array) {
    sum += n;
  }
  if (sum != Util::CharsLen(concatenated_key)) {
    return false;
  }

  // Pop back all the segments in [start_segment_index, segments_size).
  while (segments->segments_size() > start_segment_index) {
    segments->pop_back_segment();
  }

  // Create new segments using new_size_array.
  size_t pos = 0;
  for (const uint8_t new_size : new_size_array) {
    Segment* seg = segments->add_segment();
    seg->set_key(Util::Utf8SubString(concatenated_key, pos, new_size));
    seg->set_segment_type(Segment::FREE);
    pos += new_size;
  }

  ApplyConversion(segments, request);

  // Partially consumed key information is added to each candidate in the
  // segments as they are considered resized by user.
  for (size_t i = start_segment_index; i < segments->segments_size(); ++i) {
    MaybeSetConsumedKeySizeToSegment(Util::CharsLen(segments->segment(i).key()),
                                     segments->mutable_segment(i));
  }

  segments->set_resized(true);

  return true;
}

void Converter::ApplyConversion(Segments* segments,
                                const ConversionRequest& request) const {
  if (!immutable_converter_->ConvertForRequest(request, segments)) {
    return;
  }
  ApplyPostProcessing(request, segments);
}

void Converter::ApplyPostProcessing(const ConversionRequest& request,
                                    Segments* segments) const {
  RewriteAndSuppressCandidates(request, segments);
  TrimCandidates(request, segments);
}

void Converter::CompletePosIds(Candidate* candidate) const {
  if (candidate->lid != 0 && candidate->rid != 0) {
    return;
  }
  if (candidate->lid == 0 && candidate->rid != 0) {
    candidate->lid = candidate->rid;
  } else if (candidate->lid != 0 && candidate->rid == 0) {
    candidate->rid = candidate->lid;
  } else {
    candidate->lid = general_noun_id_;
    candidate->rid = general_noun_id_;
  }
}

void Converter::RewriteAndSuppressCandidates(const ConversionRequest& request,
                                             Segments* segments) const {
  if (request.skip_slow_rewriters()) {
    return;
  }
  const prediction::Result history_result = MakeHistoryResult(*segments);
  const ConversionRequest rewriter_req =
      ConversionRequestBuilder()
          .SetConversionRequestView(request)
          .SetHistoryResultView(history_result)
          .Build();
  rewriter_->Rewrite(rewriter_req, segments);
}

void Converter::TrimCandidates(const ConversionRequest& request,
                               Segments* segments) const {
  const size_t candidates_limit = request.max_conversion_candidates_size();
  if (candidates_limit == 0) {
    return;
  }
  for (Segment& segment : segments->conversion_segments()) {
    const size_t candidates_size = segment.candidates_size();
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
