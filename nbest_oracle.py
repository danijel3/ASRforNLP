from jiwer import wer

ref = {}
with open('sejm-audio/text') as f:
    for l in f:
        tok = l.strip().split()
        ref[tok[0]] = ' '.join(tok[1:])

all_wer = {}
with open('nbest.txt') as f:    
    for l in f:
      tok = l.strip().split()
      latname = tok[0]
      lattok = latname.split('-')
      sent = '-'.join(lattok[:-1])
      n = int(lattok[-1])
      ref_txt = ref[sent]
      hyp_txt = ' '.join(tok[1:])
      if sent not in all_wer:
          all_wer[sent] = {}
      if len(ref_txt) == 0:
          all_wer[sent][n] = float('inf')
      else:
          all_wer[sent][n] = wer(ref_txt, hyp_txt)

for sent, wers in all_wer.items():
    onebest = wers[1]
    oracle = onebest
    oraclepos = 1
    for pos, wer in wers.items():
        if wer < oracle:
            oracle = wer
            oraclepos = pos
    print(f'UTT {sent} WER {onebest:%} ORACLE {oracle:%} POS {oraclepos}')
