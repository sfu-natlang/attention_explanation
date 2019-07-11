#!/usr/bin/env python
""" Translator Class and builder """
from __future__ import print_function
import codecs
import os
import math
import time
from itertools import count
from collections import defaultdict, Counter

import torch

import onmt.model_builder
import onmt.translate.beam
import onmt.inputters as inputters
import onmt.decoders.ensemble
from onmt.translate.beam_search import BeamSearch
from onmt.translate.random_sampling import RandomSampling
from onmt.utils.misc import tile, set_random_seed, tvd, high_distance
from onmt.utils.plotting import *
from onmt.modules.copy_generator import collapse_copy_scores


TOTAL_TOKENS = 0
NOT_CHANGED_TOKENS_WITH_PERMUTE = 0
NOT_CHANGED_TOKENS_WITH_ZERO = 0
NOT_CHANGED_TOKENS_WITH_PERMUTE_NOT_CHANGED_WITH_ZERO = 0
NOT_CHANGED_TOKENS_WITH_EQUAL_WEIGHT = 0
NOT_CHANGED_TOKENS_WITH_LAST_STATE = 0
NOT_CHANGED_TOKENS_WITH_KEEP_MAX_ZERO_OUT_OTHER = 0

permute_attention = False
zero_out_attention = False
equal_weight_attention = False
last_state_attention = False
tvd_permute = False
keep_max_zero_out_other = True

#not_changed_tokens_at_all_dict = defaultdict(int)
not_changed_tokens_permute_dict = defaultdict(int)
not_changed_tokens_equal_weight_dict = defaultdict(int)
not_changed_tokens_keep_max_zero_out_other_dict = defaultdict(int)

max_att_dist_change_pairs = []


def build_translator(opt, report_score=True, logger=None, out_file=None):
    if out_file is None:
        out_file = codecs.open(opt.output, 'w+', 'utf-8')

    load_test_model = onmt.decoders.ensemble.load_test_model \
        if len(opt.models) > 1 else onmt.model_builder.load_test_model
    fields, model, model_opt = load_test_model(opt)

    scorer = onmt.translate.GNMTGlobalScorer.from_opt(opt)

    translator = Translator.from_opt(
        model,
        fields,
        opt,
        model_opt,
        global_scorer=scorer,
        out_file=out_file,
        report_score=report_score,
        logger=logger
    )
    return translator


class Translator(object):
    """Translate a batch of sentences with a saved model.

    Args:
        model (onmt.modules.NMTModel): NMT model to use for translation
        fields (dict[str, torchtext.data.Field]): A dict
            mapping each side to its list of name-Field pairs.
        src_reader (onmt.inputters.DataReaderBase): Source reader.
        tgt_reader (onmt.inputters.TextDataReader): Target reader.
        gpu (int): GPU device. Set to negative for no GPU.
        n_best (int): How many beams to wait for.
        min_length (int): See
            :class:`onmt.translate.decode_strategy.DecodeStrategy`.
        max_length (int): See
            :class:`onmt.translate.decode_strategy.DecodeStrategy`.
        beam_size (int): Number of beams.
        random_sampling_topk (int): See
            :class:`onmt.translate.random_sampling.RandomSampling`.
        random_sampling_temp (int): See
            :class:`onmt.translate.random_sampling.RandomSampling`.
        stepwise_penalty (bool): Whether coverage penalty is applied every step
            or not.
        dump_beam (bool): Debugging option.
        block_ngram_repeat (int): See
            :class:`onmt.translate.decode_strategy.DecodeStrategy`.
        ignore_when_blocking (set or frozenset): See
            :class:`onmt.translate.decode_strategy.DecodeStrategy`.
        replace_unk (bool): Replace unknown token.
        data_type (str): Source data type.
        verbose (bool): Print/log every translation.
        report_bleu (bool): Print/log Bleu metric.
        report_rouge (bool): Print/log Rouge metric.
        report_time (bool): Print/log total time/frequency.
        copy_attn (bool): Use copy attention.
        global_scorer (onmt.translate.GNMTGlobalScorer): Translation
            scoring/reranking object.
        out_file (TextIO or codecs.StreamReaderWriter): Output file.
        report_score (bool) : Whether to report scores
        logger (logging.Logger or NoneType): Logger.
    """

    def __init__(
            self,
            model,
            fields,
            src_reader,
            tgt_reader,
            gpu=-1,
            n_best=1,
            min_length=0,
            max_length=100,
            ratio=0.,
            beam_size=30,
            random_sampling_topk=1,
            random_sampling_temp=1,
            stepwise_penalty=None,
            dump_beam=False,
            block_ngram_repeat=0,
            ignore_when_blocking=frozenset(),
            replace_unk=False,
            phrase_table="",
            data_type="text",
            verbose=False,
            report_bleu=False,
            report_rouge=False,
            report_time=False,
            copy_attn=False,
            global_scorer=None,
            out_file=None,
            report_score=True,
            logger=None,
            seed=-1):
        self.model = model
        self.fields = fields
        tgt_field = dict(self.fields)["tgt"].base_field
        self._tgt_vocab = tgt_field.vocab
        self._tgt_eos_idx = self._tgt_vocab.stoi[tgt_field.eos_token]
        self._tgt_pad_idx = self._tgt_vocab.stoi[tgt_field.pad_token]
        self._tgt_bos_idx = self._tgt_vocab.stoi[tgt_field.init_token]
        self._tgt_unk_idx = self._tgt_vocab.stoi[tgt_field.unk_token]
        self._tgt_vocab_len = len(self._tgt_vocab)

        self._gpu = gpu
        self._use_cuda = gpu > -1
        self._dev = torch.device("cuda", self._gpu) \
            if self._use_cuda else torch.device("cpu")

        self.n_best = n_best
        self.max_length = max_length

        self.beam_size = beam_size
        self.random_sampling_temp = random_sampling_temp
        self.sample_from_topk = random_sampling_topk

        self.min_length = min_length
        self.ratio = ratio
        self.stepwise_penalty = stepwise_penalty
        self.dump_beam = dump_beam
        self.block_ngram_repeat = block_ngram_repeat
        self.ignore_when_blocking = ignore_when_blocking
        self._exclusion_idxs = {
            self._tgt_vocab.stoi[t] for t in self.ignore_when_blocking}
        self.src_reader = src_reader
        self.tgt_reader = tgt_reader
        self.replace_unk = replace_unk
        if self.replace_unk and not self.model.decoder.attentional:
            raise ValueError(
                "replace_unk requires an attentional decoder.")
        self.phrase_table = phrase_table
        self.data_type = data_type
        self.verbose = verbose
        self.report_bleu = report_bleu
        self.report_rouge = report_rouge
        self.report_time = report_time

        self.copy_attn = copy_attn

        self.global_scorer = global_scorer
        if self.global_scorer.has_cov_pen and \
                not self.model.decoder.attentional:
            raise ValueError(
                "Coverage penalty requires an attentional decoder.")
        self.out_file = out_file
        self.report_score = report_score
        self.logger = logger

        self.use_filter_pred = False
        self._filter_pred = None

        # for debugging
        self.beam_trace = self.dump_beam != ""
        self.beam_accum = None
        if self.beam_trace:
            self.beam_accum = {
                "predicted_ids": [],
                "beam_parent_ids": [],
                "scores": [],
                "log_probs": []}

        set_random_seed(seed, self._use_cuda)


        # For attention explanation experiments
        self.tvd_tokens = defaultdict(int) # max_attention > 0.5 and change prob < 0.2

    @classmethod
    def from_opt(
            cls,
            model,
            fields,
            opt,
            model_opt,
            global_scorer=None,
            out_file=None,
            report_score=True,
            logger=None):
        """Alternate constructor.

        Args:
            model (onmt.modules.NMTModel): See :func:`__init__()`.
            fields (dict[str, torchtext.data.Field]): See
                :func:`__init__()`.
            opt (argparse.Namespace): Command line options
            model_opt (argparse.Namespace): Command line options saved with
                the model checkpoint.
            global_scorer (onmt.translate.GNMTGlobalScorer): See
                :func:`__init__()`..
            out_file (TextIO or codecs.StreamReaderWriter): See
                :func:`__init__()`.
            report_score (bool) : See :func:`__init__()`.
            logger (logging.Logger or NoneType): See :func:`__init__()`.
        """

        src_reader = inputters.str2reader[opt.data_type].from_opt(opt)
        tgt_reader = inputters.str2reader["text"].from_opt(opt)
        return cls(
            model,
            fields,
            src_reader,
            tgt_reader,
            gpu=opt.gpu,
            n_best=opt.n_best,
            min_length=opt.min_length,
            max_length=opt.max_length,
            ratio=opt.ratio,
            beam_size=opt.beam_size,
            random_sampling_topk=opt.random_sampling_topk,
            random_sampling_temp=opt.random_sampling_temp,
            stepwise_penalty=opt.stepwise_penalty,
            dump_beam=opt.dump_beam,
            block_ngram_repeat=opt.block_ngram_repeat,
            ignore_when_blocking=set(opt.ignore_when_blocking),
            replace_unk=opt.replace_unk,
            phrase_table=opt.phrase_table,
            data_type=opt.data_type,
            verbose=opt.verbose,
            report_bleu=opt.report_bleu,
            report_rouge=opt.report_rouge,
            report_time=opt.report_time,
            copy_attn=model_opt.copy_attn,
            global_scorer=global_scorer,
            out_file=out_file,
            report_score=report_score,
            logger=logger,
            seed=opt.seed)

    def _log(self, msg):
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)

    def _gold_score(self, batch, memory_bank, src_lengths, src_vocabs,
                    use_src_map, enc_states, batch_size, src):
        if "tgt" in batch.__dict__:
            gs = self._score_target(
                batch, memory_bank, src_lengths, src_vocabs,
                batch.src_map if use_src_map else None)

            self.model.decoder.init_state(src, memory_bank, enc_states)
        else:
            gs = [0] * batch_size
        return gs

    def translate(
            self,
            src,
            tgt=None,
            src_dir=None,
            batch_size=None,
            attn_debug=False,
            phrase_table=""):
        """Translate content of ``src`` and get gold scores from ``tgt``.

        Args:
            src: See :func:`self.src_reader.read()`.
            tgt: See :func:`self.tgt_reader.read()`.
            src_dir: See :func:`self.src_reader.read()` (only relevant
                for certain types of data).
            batch_size (int): size of examples per mini-batch
            attn_debug (bool): enables the attention logging

        Returns:
            (`list`, `list`)

            * all_scores is a list of `batch_size` lists of `n_best` scores
            * all_predictions is a list of `batch_size` lists
                of `n_best` predictions
        """

        if batch_size is None:
            raise ValueError("batch_size must be set")

        data = inputters.Dataset(
            self.fields,
            readers=([self.src_reader, self.tgt_reader]
                     if tgt else [self.src_reader]),
            data=[("src", src), ("tgt", tgt)] if tgt else [("src", src)],
            dirs=[src_dir, None] if tgt else [src_dir],
            sort_key=inputters.str2sortkey[self.data_type],
            filter_pred=self._filter_pred
        )

        data_iter = inputters.OrderedIterator(
            dataset=data,
            device=self._dev,
            batch_size=batch_size,
            train=False,
            sort=False,
            sort_within_batch=True,
            shuffle=False
        )

        xlation_builder = onmt.translate.TranslationBuilder(
            data, self.fields, self.n_best, self.replace_unk, tgt,
            self.phrase_table
        )

        # Statistics
        counter = count(1)
        pred_score_total, pred_words_total = 0, 0
        gold_score_total, gold_words_total = 0, 0

        all_scores = []
        all_predictions = []

        start_time = time.time()

        for batch in data_iter:
            batch_data = self.translate_batch(
                batch, data.src_vocabs, attn_debug
            )
            translations = xlation_builder.from_batch(batch_data)

            for trans in translations:
                all_scores += [trans.pred_scores[:self.n_best]]
                pred_score_total += trans.pred_scores[0]
                pred_words_total += len(trans.pred_sents[0])
                if tgt is not None:
                    gold_score_total += trans.gold_score
                    gold_words_total += len(trans.gold_sent) + 1

                n_best_preds = [" ".join(pred)
                                for pred in trans.pred_sents[:self.n_best]]
                all_predictions += [n_best_preds]
                self.out_file.write('\n'.join(n_best_preds) + '\n')
                self.out_file.flush()

                if self.verbose:
                    sent_number = next(counter)
                    output = trans.log(sent_number)
                    if self.logger:
                        self.logger.info(output)
                    else:
                        os.write(1, output.encode('utf-8'))

                if attn_debug:
                    preds = trans.pred_sents[0]
                    preds.append('</s>')
                    attns = trans.attns[0].tolist()
                    if self.data_type == 'text':
                        srcs = trans.src_raw
                    else:
                        srcs = [str(item) for item in range(len(attns[0]))]
                    header_format = "{:>10.10} " + "{:>10.7} " * len(srcs)
                    row_format = "{:>10.10} " + "{:>10.7f} " * len(srcs)
                    output = header_format.format("", *srcs) + '\n'
                    for word, row in zip(preds, attns):
                        max_index = row.index(max(row))
                        row_format = row_format.replace(
                            "{:>10.7f} ", "{:*>10.7f} ", max_index + 1)
                        row_format = row_format.replace(
                            "{:*>10.7f} ", "{:>10.7f} ", max_index)
                        output += row_format.format(word, *row) + '\n'
                        row_format = "{:>10.10} " + "{:>10.7f} " * len(srcs)
                    if self.logger:
                        self.logger.info(output)
                    else:
                        os.write(1, output.encode('utf-8'))

        end_time = time.time()

        if self.report_score:
            msg = self._report_score('PRED', pred_score_total,
                                     pred_words_total)
            self._log(msg)
            if tgt is not None:
                msg = self._report_score('GOLD', gold_score_total,
                                         gold_words_total)
                self._log(msg)
                if self.report_bleu:
                    msg = self._report_bleu(tgt)
                    self._log(msg)
                if self.report_rouge:
                    msg = self._report_rouge(tgt)
                    self._log(msg)

        if self.report_time:
            total_time = end_time - start_time
            self._log("Total translation time (s): %f" % total_time)
            self._log("Average translation time (s): %f" % (
                total_time / len(all_predictions)))
            self._log("Tokens per second: %f" % (
                pred_words_total / total_time))

        if self.dump_beam:
            import json
            json.dump(self.translator.beam_accum,
                      codecs.open(self.dump_beam, 'w', 'utf-8'))

        print("TOTAL_TOKENS:  %d" % (TOTAL_TOKENS))

        if permute_attention is True:
            print("NOT_CHANGED_TOKENS_WITH_PERMUTE:  %d - ratio: %f" % (NOT_CHANGED_TOKENS_WITH_PERMUTE, NOT_CHANGED_TOKENS_WITH_PERMUTE / float(TOTAL_TOKENS)))

            print("dict:  ")
            d = Counter(not_changed_tokens_permute_dict)
            print(d.most_common(n=100))

        if zero_out_attention is True:
            print("NOT_CHANGED_TOKENS_WITH_ZERO:  %d - ratio: %f" % (NOT_CHANGED_TOKENS_WITH_ZERO, NOT_CHANGED_TOKENS_WITH_ZERO / float(TOTAL_TOKENS)))

        if permute_attention is True and zero_out_attention is True:
            print("NOT CHANGED AT ALL:  %d - ratio: %f" % (NOT_CHANGED_TOKENS_WITH_PERMUTE_NOT_CHANGED_WITH_ZERO, NOT_CHANGED_TOKENS_WITH_PERMUTE_NOT_CHANGED_WITH_ZERO / float(TOTAL_TOKENS)))

            ratio = (NOT_CHANGED_TOKENS_WITH_PERMUTE - NOT_CHANGED_TOKENS_WITH_PERMUTE_NOT_CHANGED_WITH_ZERO) / NOT_CHANGED_TOKENS_WITH_PERMUTE
            print("Ratio of tokens that not changed with permute but changed with zero:  %f" % ratio)

        if equal_weight_attention is True:
            print("NOT_CHANGED_TOKENS_WITH_EQUAL_WEIGHT:  %d - ratio: %f" % (NOT_CHANGED_TOKENS_WITH_EQUAL_WEIGHT, NOT_CHANGED_TOKENS_WITH_EQUAL_WEIGHT / float(TOTAL_TOKENS)))

            print("dict:  ")
            d = Counter(not_changed_tokens_equal_weight_dict)
            print(d.most_common(n=100))

        if last_state_attention is True:
            print("NOT_CHANGED_TOKENS_WITH_LAST_STATE:  %d - ratio: %f" % (NOT_CHANGED_TOKENS_WITH_LAST_STATE, NOT_CHANGED_TOKENS_WITH_LAST_STATE / float(TOTAL_TOKENS)))

        if keep_max_zero_out_other:
            print("NOT_CHANGED_TOKENS_WITH_KEEP_MAX_ZERO_OUT_OTHER:  %d - ratio: %f" % (NOT_CHANGED_TOKENS_WITH_KEEP_MAX_ZERO_OUT_OTHER, NOT_CHANGED_TOKENS_WITH_KEEP_MAX_ZERO_OUT_OTHER / float(TOTAL_TOKENS)))

            print("dict:  ")
            d = Counter(not_changed_tokens_keep_max_zero_out_other_dict)
            print(d.most_common(n=200))


        if tvd_permute is True:
            print("dict:  ")
            d = Counter(self.tvd_tokens)
            print(d.most_common(n=100))

            print("sum:  ")
            print(sum(self.tvd_tokens.values()))

            fig, ax = init_gridspec(3, 3, 1)

            max_attn = [el[0] for el in max_att_dist_change_pairs]
            med_diff = [el[1] for el in max_att_dist_change_pairs]
            yhat = [0] * len(max_attn)

            plot_violin_by_class(ax[0], max_attn, med_diff, yhat, xlim=(0, 1.0))
            annotate(ax[0], xlim=(-0.05, 1.05), ylabel="Max attention", xlabel="Median Output Difference", legend=None)

            adjust_gridspec()
            save_axis_in_file(fig, ax[0], "/cs/natlang-expts/pooya/attention_explanation/plots", "tvd_permutation")
            show_gridspec()

            print("len:  %d", len(max_att_dist_change_pairs))

        return all_scores, all_predictions

    def _translate_random_sampling(
            self,
            batch,
            src_vocabs,
            max_length,
            min_length=0,
            sampling_temp=1.0,
            keep_topk=-1,
            return_attention=False):
        """Alternative to beam search. Do random sampling at each step."""

        assert self.beam_size == 1

        # TODO: support these blacklisted features.
        assert self.block_ngram_repeat == 0

        batch_size = batch.batch_size

        # Encoder forward.
        src, enc_states, memory_bank, src_lengths = self._run_encoder(batch)
        self.model.decoder.init_state(src, memory_bank, enc_states)

        use_src_map = self.copy_attn

        results = {
            "predictions": None,
            "scores": None,
            "attention": None,
            "batch": batch,
            "gold_score": self._gold_score(
                batch, memory_bank, src_lengths, src_vocabs, use_src_map,
                enc_states, batch_size, src)}

        memory_lengths = src_lengths
        src_map = batch.src_map if use_src_map else None

        if isinstance(memory_bank, tuple):
            mb_device = memory_bank[0].device
        else:
            mb_device = memory_bank.device

        random_sampler = RandomSampling(
            self._tgt_pad_idx, self._tgt_bos_idx, self._tgt_eos_idx,
            batch_size, mb_device, min_length, self.block_ngram_repeat,
            self._exclusion_idxs, return_attention, self.max_length,
            sampling_temp, keep_topk, memory_lengths)

        for step in range(max_length):
            # Shape: (1, B, 1)
            decoder_input = random_sampler.alive_seq[:, -1].view(1, -1, 1)

            log_probs, attn, hack_dict = self._decode_and_generate(
                decoder_input,
                memory_bank,
                batch,
                src_vocabs,
                memory_lengths=memory_lengths,
                src_map=src_map,
                step=step,
                batch_offset=random_sampler.select_indices,
                permute_attention=permute_attention,
                zero_out_attention=zero_out_attention,
                equal_weight_attention=equal_weight_attention,
                last_state_attention=last_state_attention,
                tvd_permute=tvd_permute,
                keep_max_zero_out_other=keep_max_zero_out_other
            )

            top_prob = torch.topk(log_probs, k=1, dim=1)


            global TOTAL_TOKENS
            global NOT_CHANGED_TOKENS_WITH_PERMUTE
            global NOT_CHANGED_TOKENS_WITH_ZERO
            global NOT_CHANGED_TOKENS_WITH_EQUAL_WEIGHT
            global NOT_CHANGED_TOKENS_WITH_LAST_STATE
            global NOT_CHANGED_TOKENS_WITH_PERMUTE_NOT_CHANGED_WITH_ZERO
            global not_changed_tokens_permute_dict
            global not_changed_tokens_equal_weight_dict
            global max_att_dist_change_pairs

            global NOT_CHANGED_TOKENS_WITH_KEEP_MAX_ZERO_OUT_OTHER
            global not_changed_tokens_keep_max_zero_out_other_dict

            TOTAL_TOKENS += top_prob.indices.size()[0]

            tgt_field = dict(self.fields)["tgt"].base_field
            vocab = tgt_field.vocab

            if permute_attention is True:
                log_probs_permute_attention = hack_dict['log_probs_permute_attention']
                top_prob_permute = torch.topk(log_probs_permute_attention, k=1, dim=1)
                equality_permute = (top_prob.indices == top_prob_permute.indices)
                equality_permute_cpu = equality_permute.cpu()

                NOT_CHANGED_TOKENS_WITH_PERMUTE += equality_permute.sum(dim=0).cpu().numpy()[0]

                for i in range(equality_permute.size()[0]):
                    if(equality_permute_cpu[i][0] == 1):
                        not_changed_tokens_permute_dict[vocab.itos[top_prob.indices[i][0]]] += 1

            if zero_out_attention is True:
                log_probs_zero_out_attention = hack_dict['log_probs_zero_out_attention']
                top_prob_zero = torch.topk(log_probs_zero_out_attention, k=1, dim=1)
                equality_zero = (top_prob.indices == top_prob_zero.indices)

                NOT_CHANGED_TOKENS_WITH_ZERO += equality_zero.sum(dim=0).cpu().numpy()[0]

            if permute_attention is True and zero_out_attention is True:
                not_changed_at_all = 0
                for i in range(equality_permute.size()[0]):
                    if(equality_permute_cpu[i][0] == 1 and equality_zero_cpu[i][0] == 1):
                        not_changed_at_all += 1

                NOT_CHANGED_TOKENS_WITH_PERMUTE_NOT_CHANGED_WITH_ZERO += not_changed_at_all

            if equal_weight_attention is True:
                log_probs_equal_weight_attention = hack_dict['log_probs_equal_weight_attention']
                top_prob_equal_weight = torch.topk(log_probs_equal_weight_attention, k=1, dim=1)
                equality_equal_weight = (top_prob.indices == top_prob_equal_weight.indices)
                equality_equal_weight_cpu = equality_equal_weight.cpu()

                NOT_CHANGED_TOKENS_WITH_EQUAL_WEIGHT += equality_equal_weight.sum(dim=0).cpu().numpy()[0]

                for i in range(equality_equal_weight.size()[0]):
                    if(equality_equal_weight_cpu[i][0] == 1):
                        not_changed_tokens_equal_weight_dict[vocab.itos[top_prob.indices[i][0]]] += 1

            if last_state_attention is True:
                log_probs_last_state_attention = hack_dict['log_probs_last_state_attention']
                top_prob_last_state = torch.topk(log_probs_last_state_attention, k=1, dim=1)
                equality_last_state = (top_prob.indices == top_prob_last_state.indices)

                NOT_CHANGED_TOKENS_WITH_LAST_STATE += equality_last_state.sum(dim=0).cpu().numpy()[0]

            if keep_max_zero_out_other is True:
                log_probs_keep_max_zero_out_other_attention = hack_dict['log_probs_keep_max_zero_out_other_attention']
                top_prob_keep_max_zero_out_other = torch.topk(log_probs_keep_max_zero_out_other_attention, k=1, dim=1)
                equality_keep_max_zero_out_other = (top_prob.indices == top_prob_keep_max_zero_out_other.indices)
                equality_keep_max_zero_out_other_cpu = equality_keep_max_zero_out_other.cpu()

                NOT_CHANGED_TOKENS_WITH_KEEP_MAX_ZERO_OUT_OTHER += equality_keep_max_zero_out_other.sum(dim=0).cpu().numpy()[0]

                for i in range(equality_keep_max_zero_out_other.size()[0]):
                    if(equality_keep_max_zero_out_other_cpu[i][0] == 1):
                        not_changed_tokens_keep_max_zero_out_other_dict[vocab.itos[top_prob.indices[i][0]]] += 1



            if tvd_permute is True:
                max_attention = hack_dict['tvd_max_attention'].cpu()
                dist_change_median = hack_dict['tvd_dist_change_median'].cpu()

                for i in range(top_prob.indices.size()[0]):
                    if(max_attention[i] > 0.5 and dist_change_median[i] < 0.2):
                        self.tvd_tokens[vocab.itos[top_prob.indices[i][0]]] += 1

                assert (len(max_attention.size()) == 1)
                assert (len(dist_change_median.size()) == 1)
                assert (max_attention.size()[0] == dist_change_median.size()[0])

                for i in range(max_attention.size()[0]):
                    max_att_dist_change_pairs.append((float(max_attention[i].cpu()), float(dist_change_median[i].cpu())))


            random_sampler.advance(log_probs, attn)
            any_batch_is_finished = random_sampler.is_finished.any()
            if any_batch_is_finished:
                random_sampler.update_finished()
                if random_sampler.done:
                    break

            if any_batch_is_finished:
                select_indices = random_sampler.select_indices

                # Reorder states.
                if isinstance(memory_bank, tuple):
                    memory_bank = tuple(x.index_select(1, select_indices)
                                        for x in memory_bank)
                else:
                    memory_bank = memory_bank.index_select(1, select_indices)

                memory_lengths = memory_lengths.index_select(0, select_indices)

                if src_map is not None:
                    src_map = src_map.index_select(1, select_indices)

                self.model.decoder.map_state(
                    lambda state, dim: state.index_select(dim, select_indices))

        results["scores"] = random_sampler.scores
        results["predictions"] = random_sampler.predictions
        results["attention"] = random_sampler.attention
        return results

    def translate_batch(self, batch, src_vocabs, attn_debug):
        """Translate a batch of sentences."""
        with torch.no_grad():
            if self.beam_size == 1:
                return self._translate_random_sampling(
                    batch,
                    src_vocabs,
                    self.max_length,
                    min_length=self.min_length,
                    sampling_temp=self.random_sampling_temp,
                    keep_topk=self.sample_from_topk,
                    return_attention=attn_debug or self.replace_unk)
            else:
                return self._translate_batch(
                    batch,
                    src_vocabs,
                    self.max_length,
                    min_length=self.min_length,
                    ratio=self.ratio,
                    n_best=self.n_best,
                    return_attention=attn_debug or self.replace_unk)

    def _run_encoder(self, batch):
        src, src_lengths = batch.src if isinstance(batch.src, tuple) \
                           else (batch.src, None)

        enc_states, memory_bank, src_lengths = self.model.encoder(
            src, src_lengths)
        if src_lengths is None:
            assert not isinstance(memory_bank, tuple), \
                'Ensemble decoding only supported for text data'
            src_lengths = torch.Tensor(batch.batch_size) \
                               .type_as(memory_bank) \
                               .long() \
                               .fill_(memory_bank.size(0))
        return src, enc_states, memory_bank, src_lengths

    def _decode_and_generate(
            self,
            decoder_in,
            memory_bank,
            batch,
            src_vocabs,
            memory_lengths,
            src_map=None,
            step=None,
            batch_offset=None,
            permute_attention=False,
            zero_out_attention=False,
            equal_weight_attention=False,
            last_state_attention=False,
            keep_max_zero_out_other=False,
            tvd_permute=False):

        if self.copy_attn:
            # Turn any copied words into UNKs.
            decoder_in = decoder_in.masked_fill(
                decoder_in.gt(self._tgt_vocab_len - 1), self._tgt_unk_idx
            )

        # Decoder forward, takes [tgt_len, batch, nfeats] as input
        # and [src_len, batch, hidden] as memory_bank
        # in case of inference tgt_len = 1, batch = beam times batch_size
        # in case of Gold Scoring tgt_len = actual length, batch = 1 batch

        #print(">>>>> hidden state of the decoder:    <<<<<")
        #print(self.model.decoder.state["hidden"][1])
        #before_state1, before_state2 = self.model.decoder.state["hidden"]]0].clone(), self.model.decoder.state["hidden"][1].clone()
        #print(before_state == self.model.decoder.state["hidden"])

        dec_out, dec_attn = self.model.decoder(
            decoder_in, memory_bank, memory_lengths=memory_lengths, step=step, permute_attention=permute_attention, zero_out_attention=zero_out_attention, equal_weight_attention=equal_weight_attention,
            last_state_attention=last_state_attention, tvd_permute=tvd_permute, keep_max_zero_out_other=keep_max_zero_out_other
        )

        hack_dict = {}

        # Generator forward.
        if not self.copy_attn:
            if "std" in dec_attn:
                attn = dec_attn["std"]
            else:
                attn = None

            log_probs = self.model.generator(dec_out.squeeze(0))

            if permute_attention is True:
                hack_dict['log_probs_permute_attention'] = self.model.generator(dec_attn["std_permute"][1].squeeze(0))

            if zero_out_attention is True:
                hack_dict['log_probs_zero_out_attention'] = self.model.generator(dec_attn["std_zero_out"][1].squeeze(0))

            if equal_weight_attention is True:
                hack_dict['log_probs_equal_weight_attention'] = self.model.generator(dec_attn["std_equal_weight"][1].squeeze(0))

            if last_state_attention is True:
                hack_dict['log_probs_last_state_attention'] = self.model.generator(dec_attn["std_last_state"][1].squeeze(0))

            if keep_max_zero_out_other is True:
                hack_dict['log_probs_keep_max_zero_out_other_attention'] = self.model.generator(dec_attn["std_keep_max_zero_out_other"][1].squeeze(0))

            if tvd_permute is True:
                dec_outs = dec_attn["std_tvd_permute"]
                distances = []
                for my_dec_out in dec_outs:
                    my_dec_out = my_dec_out.squeeze(0)
                    my_log_prob = self.model.generator(my_dec_out)

                    #distance = tvd(torch.exp(log_probs), torch.exp(my_log_prob))
                    distance = high_distance(torch.exp(log_probs), torch.exp(my_log_prob))[0]
                    distances.append(distance)

                dist_change_median = torch.median(torch.stack(distances), dim=0).values

                if attn.size()[0] != 1:
                    print(">>>> Shit! Target length in attention is more than 1 <<<<")
                    assert False

                max_attention = attn[0].max(dim=1).values

                hack_dict['tvd_dist_change_median'] = dist_change_median
                hack_dict['tvd_max_attention'] = max_attention

            # returns [(batch_size x beam_size) , vocab ] when 1 step
            # or [ tgt_len, batch_size, vocab ] when full sentence
        else:
            attn = dec_attn["copy"]
            scores = self.model.generator(dec_out.view(-1, dec_out.size(2)),
                                          attn.view(-1, attn.size(2)),
                                          src_map)
            # here we have scores [tgt_lenxbatch, vocab] or [beamxbatch, vocab]
            if batch_offset is None:
                scores = scores.view(batch.batch_size, -1, scores.size(-1))
            else:
                scores = scores.view(-1, self.beam_size, scores.size(-1))
            scores = collapse_copy_scores(
                scores,
                batch,
                self._tgt_vocab,
                src_vocabs,
                batch_dim=0,
                batch_offset=batch_offset
            )
            scores = scores.view(decoder_in.size(0), -1, scores.size(-1))
            log_probs = scores.squeeze(0).log()
            # returns [(batch_size x beam_size) , vocab ] when 1 step
            # or [ tgt_len, batch_size, vocab ] when full sentence

        if any([permute_attention, zero_out_attention, equal_weight_attention, last_state_attention, tvd_permute, keep_max_zero_out_other]):
            return log_probs, attn, hack_dict
        else:
            return log_probs, attn

    def _translate_batch(
            self,
            batch,
            src_vocabs,
            max_length,
            min_length=0,
            ratio=0.,
            n_best=1,
            return_attention=False):
        # TODO: support these blacklisted features.
        assert not self.dump_beam

        # (0) Prep the components of the search.
        use_src_map = self.copy_attn
        beam_size = self.beam_size
        batch_size = batch.batch_size

        #print(">>>> batch:   <<<<<<")
        #print("type of batch:  ")
        #print(type(batch))
        #print("batch dimension:  ")
        #print(batch)
        #print("dimension for src:  ")
        #print(batch.src[0].size())
        #print(batch.src[0][:,0,:])
        #print(batch.src[1][0])
        #print("=====================")
        # (1) Run the encoder on the src.
        src, enc_states, memory_bank, src_lengths = self._run_encoder(batch)
        self.model.decoder.init_state(src, memory_bank, enc_states)

        results = {
            "predictions": None,
            "scores": None,
            "attention": None,
            "batch": batch,
            "gold_score": self._gold_score(
                batch, memory_bank, src_lengths, src_vocabs, use_src_map,
                enc_states, batch_size, src)}

        # (2) Repeat src objects `beam_size` times.
        # We use batch_size x beam_size

        src_map = (tile(batch.src_map, beam_size, dim=1)
                   if use_src_map else None)
        self.model.decoder.map_state(
            lambda state, dim: tile(state, beam_size, dim=dim))

        if isinstance(memory_bank, tuple):
            memory_bank = tuple(tile(x, beam_size, dim=1) for x in memory_bank)
            mb_device = memory_bank[0].device
        else:
            memory_bank = tile(memory_bank, beam_size, dim=1)
            mb_device = memory_bank.device
        memory_lengths = tile(src_lengths, beam_size)

        # (0) pt 2, prep the beam object
        beam = BeamSearch(
            beam_size,
            n_best=n_best,
            batch_size=batch_size,
            global_scorer=self.global_scorer,
            pad=self._tgt_pad_idx,
            eos=self._tgt_eos_idx,
            bos=self._tgt_bos_idx,
            min_length=min_length,
            ratio=ratio,
            max_length=max_length,
            mb_device=mb_device,
            return_attention=return_attention,
            stepwise_penalty=self.stepwise_penalty,
            block_ngram_repeat=self.block_ngram_repeat,
            exclusion_tokens=self._exclusion_idxs,
            memory_lengths=memory_lengths)

        for step in range(max_length):
            decoder_input = beam.current_predictions.view(1, -1, 1)

            log_probs, attn = self._decode_and_generate(
                decoder_input,
                memory_bank,
                batch,
                src_vocabs,
                memory_lengths=memory_lengths,
                src_map=src_map,
                step=step,
                batch_offset=beam._batch_offset)

            beam.advance(log_probs, attn)
            any_beam_is_finished = beam.is_finished.any()
            if any_beam_is_finished:
                beam.update_finished()
                if beam.done:
                    break

            select_indices = beam.current_origin

            if any_beam_is_finished:
                # Reorder states.
                if isinstance(memory_bank, tuple):
                    memory_bank = tuple(x.index_select(1, select_indices)
                                        for x in memory_bank)
                else:
                    memory_bank = memory_bank.index_select(1, select_indices)

                memory_lengths = memory_lengths.index_select(0, select_indices)

                if src_map is not None:
                    src_map = src_map.index_select(1, select_indices)

            self.model.decoder.map_state(
                lambda state, dim: state.index_select(dim, select_indices))

        results["scores"] = beam.scores
        results["predictions"] = beam.predictions
        results["attention"] = beam.attention
        return results

    # This is left in the code for now, but unsued
    def _translate_batch_deprecated(self, batch, src_vocabs):
        # (0) Prep each of the components of the search.
        # And helper method for reducing verbosity.
        use_src_map = self.copy_attn
        beam_size = self.beam_size
        batch_size = batch.batch_size

        beam = [onmt.translate.Beam(
            beam_size,
            n_best=self.n_best,
            cuda=self.cuda,
            global_scorer=self.global_scorer,
            pad=self._tgt_pad_idx,
            eos=self._tgt_eos_idx,
            bos=self._tgt_bos_idx,
            min_length=self.min_length,
            stepwise_penalty=self.stepwise_penalty,
            block_ngram_repeat=self.block_ngram_repeat,
            exclusion_tokens=self._exclusion_idxs)
            for __ in range(batch_size)]

        # (1) Run the encoder on the src.
        src, enc_states, memory_bank, src_lengths = self._run_encoder(batch)
        self.model.decoder.init_state(src, memory_bank, enc_states)

        results = {
            "predictions": [],
            "scores": [],
            "attention": [],
            "batch": batch,
            "gold_score": self._gold_score(
                batch, memory_bank, src_lengths, src_vocabs, use_src_map,
                enc_states, batch_size, src)}

        # (2) Repeat src objects `beam_size` times.
        # We use now  batch_size x beam_size (same as fast mode)
        src_map = (tile(batch.src_map, beam_size, dim=1)
                   if use_src_map else None)
        self.model.decoder.map_state(
            lambda state, dim: tile(state, beam_size, dim=dim))

        if isinstance(memory_bank, tuple):
            memory_bank = tuple(tile(x, beam_size, dim=1) for x in memory_bank)
        else:
            memory_bank = tile(memory_bank, beam_size, dim=1)
        memory_lengths = tile(src_lengths, beam_size)

        # (3) run the decoder to generate sentences, using beam search.
        for i in range(self.max_length):
            if all((b.done for b in beam)):
                break

            # (a) Construct batch x beam_size nxt words.
            # Get all the pending current beam words and arrange for forward.

            inp = torch.stack([b.current_predictions for b in beam])
            inp = inp.view(1, -1, 1)

            # (b) Decode and forward
            out, beam_attn = self._decode_and_generate(
                inp, memory_bank, batch, src_vocabs,
                memory_lengths=memory_lengths, src_map=src_map, step=i
            )
            out = out.view(batch_size, beam_size, -1)
            beam_attn = beam_attn.view(batch_size, beam_size, -1)

            # (c) Advance each beam.
            select_indices_array = []
            # Loop over the batch_size number of beam
            for j, b in enumerate(beam):
                if not b.done:
                    b.advance(out[j, :],
                              beam_attn.data[j, :, :memory_lengths[j]])
                select_indices_array.append(
                    b.current_origin + j * beam_size)
            select_indices = torch.cat(select_indices_array)

            self.model.decoder.map_state(
                lambda state, dim: state.index_select(dim, select_indices))

        # (4) Extract sentences from beam.
        for b in beam:
            scores, ks = b.sort_finished(minimum=self.n_best)
            hyps, attn = [], []
            for times, k in ks[:self.n_best]:
                hyp, att = b.get_hyp(times, k)
                hyps.append(hyp)
                attn.append(att)
            results["predictions"].append(hyps)
            results["scores"].append(scores)
            results["attention"].append(attn)

        return results

    def _score_target(self, batch, memory_bank, src_lengths,
                      src_vocabs, src_map):
        tgt = batch.tgt
        tgt_in = tgt[:-1]

        log_probs, attn = self._decode_and_generate(
            tgt_in, memory_bank, batch, src_vocabs,
            memory_lengths=src_lengths, src_map=src_map)

        log_probs[:, :, self._tgt_pad_idx] = 0
        gold = tgt[1:]
        gold_scores = log_probs.gather(2, gold)
        gold_scores = gold_scores.sum(dim=0).view(-1)

        return gold_scores

    def _report_score(self, name, score_total, words_total):
        if words_total == 0:
            msg = "%s No words predicted" % (name,)
        else:
            msg = ("%s AVG SCORE: %.4f, %s PPL: %.4f" % (
                name, score_total / words_total,
                name, math.exp(-score_total / words_total)))
        return msg

    def _report_bleu(self, tgt_path):
        import subprocess
        base_dir = os.path.abspath(__file__ + "/../../..")
        # Rollback pointer to the beginning.
        self.out_file.seek(0)
        print()

        res = subprocess.check_output(
            "perl %s/tools/multi-bleu.perl %s" % (base_dir, tgt_path),
            stdin=self.out_file, shell=True
        ).decode("utf-8")

        msg = ">> " + res.strip()
        return msg

    def _report_rouge(self, tgt_path):
        import subprocess
        path = os.path.split(os.path.realpath(__file__))[0]
        msg = subprocess.check_output(
            "python %s/tools/test_rouge.py -r %s -c STDIN" % (path, tgt_path),
            shell=True, stdin=self.out_file
        ).decode("utf-8").strip()
        return msg
