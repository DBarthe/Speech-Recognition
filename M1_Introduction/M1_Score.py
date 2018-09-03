import argparse
import wer
import re

# create a function that calls wer.string_edit_distance() on every utterance
# and accumulates the errors for the corpus. Then, report the word error rate (WER)
# and the sentence error rate (SER). The WER should include the the total errors as well as the
# separately reporting the percentage of insertions, deletions and substitutions.
# The function signature is
# num_tokens, num_errors, num_deletions, num_insertions, num_substitutions = wer.string_edit_distance(ref=reference_string, hyp=hypothesis_string)
#
def score(ref_trn=None, hyp_trn=None):

    wer_sum = wer_err = wer_del = wer_ins = wer_sub = 0
    ser_sum = ser_err = 0

    for ref_line, hyp_line in zip(open(ref_trn), open(hyp_trn)):
        ref_m = re.search('(.*) \((.*)\)', ref_line)
        hyp_m = re.search('(.*) \((.*)\)', hyp_line)
        track_id = ref_m.group(2)
        ref_string = ref_m.group(1).split()
        hyp_string = hyp_m.group(1).split()

        print ("id: ({})".format(track_id))
        print("ref: {}".format(ref_string))
        print("hyp: {}".format(hyp_string))

        num_tokens, num_errors, num_deletions, num_insertions, num_substitutions = wer.string_edit_distance(ref=ref_string, hyp=hyp_string)
        print ("Score N={} S={} D={} I={}".format(num_tokens, num_substitutions, num_deletions, num_insertions))

        wer_sum += num_tokens
        wer_err += num_errors
        wer_del += num_deletions
        wer_sub += num_substitutions
        wer_ins += num_insertions

        ser_sum += 1
        ser_err += 0 if num_errors == 0 else 1

        print("")

    print ("-----------------")
    print ("Sentence Error Rate:")
    print ("Sum: N={} Err={}".format(ser_sum, ser_err))
    print ("Avg: N={} Err={:.0%}".format(ser_sum, ser_err / ser_sum))
    print ("-----------------")
    print ("Sentence Error Rate:")
    print ("Sum: N={} Err={} Sub={} Del={} Ins={}".format(wer_sum, wer_err, wer_sub, wer_del, wer_ins))
    print ("Avg: N={} Err={:.0%} Sub={:.0%} Del={:.0%} Ins={:.0%}".format(wer_sum, wer_err / wer_sum, wer_sub / wer_sum, wer_del / wer_sum, wer_ins / wer_sum))
    print ("-----------------")
    return


if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Evaluate ASR results.\n"
                                                 "Computes Word Error Rate and Sentence Error Rate")
    parser.add_argument('-ht', '--hyptrn', help='Hypothesized transcripts in TRN format', required=False, default="misc/hyp.trn")
    parser.add_argument('-rt', '--reftrn', help='Reference transcripts in TRN format', required=False, default="misc/ref.trn")
    args = parser.parse_args()

    if args.reftrn is None or args.hyptrn is None:
        RuntimeError("Must specify reference trn and hypothesis trn files.")

    score(ref_trn=args.reftrn, hyp_trn=args.hyptrn)
