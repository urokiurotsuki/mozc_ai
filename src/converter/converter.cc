// Copyright 2010-2025, Google Inc.
// All rights reserved.
//
// Mozc AI IME v7.0 - Extreme Performance (Build Ready)
//
// 設計変更点:
// - 不要なレガシープロトコル(V0/Legacy)のハンドリングを削除しV1に一本化
// - AI候補がない場合の「英語強制注入(Injection)」ロジックの実装
// - タイムアウト時の高速脱出(Early Exit)
// - TAB補完(Prediction)時にもAIを作動させるフックを追加
//
// プロトコル V1:
//   V1\t{history}\t{seg_count}\t{reading1}\t{cand_count1}\t{cand1}...\t{reading2}...

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
// AI Server Client v7.0 (Implementation)
// ==========================================

#ifdef _WIN32

// パイプ設定
constexpr char kAiPipeName[] = "\\\\.\\pipe\\MozcBertPipe";
constexpr char kAiLearnPipeName[] = "\\\\.\\pipe\\MozcAILearnPipe";
constexpr char kProtocolVersion[] = "V1";

// タイムアウト設定 (ms)
// 応答性を最優先し、80msを超えたら即座にAIを諦める
constexpr DWORD kPipeTimeoutMs = 80;
constexpr DWORD kLearnPipeTimeoutMs = 30;

// 通信量削減のため、送信する候補数は上位のみに絞る
constexpr int kMaxCandidatesToSend = 8;

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
 * AIサーバーにリランキング要求 (Strict V1)
 * 戻り値: 変更があったかどうか
 */
bool QueryAiConversionBatch(const std::string& history_context, 
                            Segments* segments) {
    if (segments->conversion_segments_size() == 0) {
        return false;
    }
    
    size_t seg_count = segments->conversion_segments_size();
    
    // ペイロード構築
    std::ostringstream payload;
    payload << kProtocolVersion << "\t";
    payload << history_context << "\t";
    payload << seg_count;
    
    for (size_t i = 0; i < seg_count; ++i) {
        const Segment& seg = segments->conversion_segment(i);
        std::string reading(seg.key().data(), seg.key().size());
        
        int cand_count = std::min(static_cast<int>(seg.candidates_size()), 
                                   kMaxCandidatesToSend);
        
        payload << "\t" << reading;
        payload << "\t" << cand_count;
        
        for (int j = 0; j < cand_count; ++j) {
            payload << "\t" << seg.candidate(j).value;
        }
    }
    payload << "\n";
    
    std::string payload_str = payload.str();
    char buffer[65536];
    DWORD bytes_read = 0;
    
    // パイプ通信 (ブロッキングだがタイムアウト付き)
    BOOL success = ::CallNamedPipeA(
        kAiPipeName,
        const_cast<char*>(payload_str.c_str()),
        static_cast<DWORD>(payload_str.size()),
        buffer,
        sizeof(buffer),
        &bytes_read,
        kPipeTimeoutMs
    );
    
    // 【高速脱出】通信失敗、タイムアウト、空応答なら即座に終了
    if (!success || bytes_read == 0) {
        return false;
    }
    
    // レスポンス解析
    std::string response(buffer, bytes_read);
    std::vector<std::string> ai_candidates;
    
    {
        std::istringstream iss(response);
        std::string token;
        while (std::getline(iss, token, '\t')) {
            if (!token.empty() && token.back() == '\n') token.pop_back();
            if (!token.empty()) ai_candidates.push_back(token);
        }
    }
    
    if (ai_candidates.empty()) return false;

    // 結果適用: AI候補の強制適用 (Injection)
    // AI候補リストの「後ろ」から順に処理し、リスト先頭へmoveしていくことで
    // 最終的にAIのTop1候補がセグメントの先頭(index 0)に来るようにする。
    
    bool changed = false;

    for (int k = static_cast<int>(ai_candidates.size()) - 1; k >= 0; --k) {
        const std::string& ai_val = ai_candidates[k];
        if (ai_val.empty()) continue;

        bool injected_or_moved = false;

        // 全セグメントを走査してマッチング
        for (size_t i = 0; i < seg_count; ++i) {
            Segment* seg = segments->mutable_conversion_segment(i);
            
            // 1. 既存候補にあるか検索
            for (size_t j = 0; j < seg->candidates_size(); ++j) {
                if (seg->candidate(j).value == ai_val) {
                    // あれば先頭へ移動 (Rank Up)
                    seg->move_candidate(static_cast<int>(j), 0);
                    seg->mutable_candidate(0)->attributes |= Attribute::RERANKED;
                    // AI推奨なのでコスト0（最高優先度）
                    seg->mutable_candidate(0)->cost = 0;
                    injected_or_moved = true;
                    changed = true;
                    break;
                }
            }
            if (injected_or_moved) break; // 1つのAI候補は1つのセグメントにのみ適用
        }

        // 2. どのセグメントにもなければ、先頭セグメントに強制追加 (Injection)
        // これにより「ういんどうず」→「Windows」等の未知語・英語変換を実現
        if (!injected_or_moved && seg_count > 0) {
            // 異常に長い候補は無視
            if (ai_val.size() > 200) continue;

            Segment* target_seg = segments->mutable_conversion_segment(0);
            
            // 新規候補作成
            Candidate* cand = target_seg->push_back_candidate();
            cand->Init();
            cand->key = target_seg->key(); 
            cand->content_key = target_seg->key();
            cand->value = ai_val;
            cand->content_value = ai_val;
            cand->cost = 0; // 最強
            cand->structure_cost = 0;
            cand->attributes |= Attribute::RERANKED;
            
            // 品詞IDの継承（安全策として既存トップからコピー）
            if (target_seg->candidates_size() > 1) {
                const Candidate& base = target_seg->candidate(0); 
                cand->lid = base.lid;
                cand->rid = base.rid;
            } else {
                cand->lid = 0;
                cand->rid = 0;
            }

            // 追加した候補(末尾)を先頭へ移動
            target_seg->move_candidate(static_cast<int>(target_seg->candidates_size()) - 1, 0);
            
            changed = true;
        }
    }
    
    return changed;
}

/**
 * 学習通知 (CommitSegments用: インデックス指定あり)
 */
void NotifyLearnToAI(const Segments& segments, 
                     absl::Span<const size_t> candidate_indices) {
    if (candidate_indices.empty()) return;
    
    std::ostringstream payload;
    payload << "LEARN";
    
    bool has_data = false;
    for (size_t i = 0; i < candidate_indices.size() && i < segments.conversion_segments_size(); ++i) {
        const Segment& seg = segments.conversion_segment(i);
        size_t idx = candidate_indices[i];
        
        if (idx < seg.candidates_size()) {
            std::string reading(seg.key().data(), seg.key().size());
            std::string value = seg.candidate(idx).value;
            payload << "\t" << reading << "\t" << value;
            has_data = true;
        }
    }
    
    if (!has_data) return;
    
    std::string payload_str = payload.str();
    char buffer[32];
    DWORD bytes_read = 0;
    
    // Fire and Forget
    ::CallNamedPipeA(
        kAiLearnPipeName,
        const_cast<char*>(payload_str.c_str()),
        static_cast<DWORD>(payload_str.size()),
        buffer,
        sizeof(buffer),
        &bytes_read,
        kLearnPipeTimeoutMs
    );
}

/**
 * 学習通知 (FinishConversion用: 先頭候補(=確定候補)を使用)
 */
void NotifyLearnToAIFromSegments(const Segments& segments) {
    if (segments.conversion_segments_size() == 0) return;
    
    std::ostringstream payload;
    payload << "LEARN";
    
    bool has_data = false;
    for (size_t i = 0; i < segments.conversion_segments_size(); ++i) {
        const Segment& seg = segments.conversion_segment(i);
        if (seg.candidates_size() > 0) {
            std::string reading(seg.key().data(), seg.key().size());
            std::string value = seg.candidate(0).value; // 確定済みなので0番目
            payload << "\t" << reading << "\t" << value;
            has_data = true;
        }
    }
    
    if (!has_data) return;
    
    std::string payload_str = payload.str();
    char buffer[32];
    DWORD bytes_read = 0;
    
    ::CallNamedPipeA(
        kAiLearnPipeName,
        const_cast<char*>(payload_str.c_str()),
        static_cast<DWORD>(payload_str.size()),
        buffer,
        sizeof(buffer),
        &bytes_read,
        kLearnPipeTimeoutMs
    );
}

#else
// Non-Windows stubs
inline void DebugLog(const char* format, ...) {}
std::string GetHistoryContext(const Segments& segments) { return ""; }
bool QueryAiConversionBatch(const std::string& history_context, Segments* segments) { return false; }
void NotifyLearnToAI(const Segments& segments, absl::Span<const size_t> candidate_indices) {}
void NotifyLearnToAIFromSegments(const Segments& segments) {}
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
  // V7: 通常変換(Spaceキー)時のAIリランキング
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
  ApplyPostProcessing(request, segments);

#ifdef _WIN32
  // V7: TAB補完(PREDICTION)時のAIフック
  // 入力中のサジェスト(PARTIAL_SUGGESTION)ではAIを呼ばないことで軽快さを維持
  if (request.request_type() == ConversionRequest::PREDICTION) {
      std::string history = GetHistoryContext(*segments);
      QueryAiConversionBatch(history, segments);
  }
#endif

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

void Converter::FinishConversion(const ConversionRequest& request,
                                 Segments* segments) const {
#ifdef _WIN32
  // V7: 確定時に学習通知 (FinishConversion用)
  NotifyLearnToAIFromSegments(*segments);
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
#ifdef _WIN32
  // V7: 部分確定時に学習通知 (インデックス指定あり)
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
  if (std::optional<RewriterInterface::ResizeSegmentsRequest> resize_request =
          rewriter_->CheckResizeSegmentsRequest(request, *segments);
      resize_request.has_value()) {
    if (ResizeSegments(segments, request, resize_request->segment_index,
                       resize_request->segment_sizes)) {
      return;
    }
  }

  if (!rewriter_->Rewrite(request, segments)) {
    return;
  }

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

}
