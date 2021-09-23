import math
import copy
from pathlib import Path
import shlex
from subprocess import run
from tempfile import NamedTemporaryFile

import openfst_python as fst


def write_nonterminal_arcs(start_state, loop_state, next_state,
                           nonterminals, left_context_phones):
    """This function relates to the grammar-decoding setup, see
    kaldi-asr.org/doc/grammar.html.  It is called from write_fst_no_silence
    and write_fst_silence, and writes to the stdout some extra arcs
    in the lexicon FST that relate to nonterminal symbols.
    See the section "Special symbols in L.fst,
    kaldi-asr.org/doc/grammar.html#grammar_special_l.
       start_state: the start-state of L.fst.
       loop_state:  the state of high out-degree in L.fst where words leave
                  and enter.
       next_state: the number from which this function can start allocating its
                  own states.  the updated value of next_state will be returned.
       nonterminals: the user-defined nonterminal symbols as a list of
          strings, e.g. ['#nonterm:contact_list', ... ].
       left_context_phones: a list of phones that may appear as left-context,
          e.g. ['a', 'ah', ... '#nonterm_bos'].
    """
    shared_state = next_state
    next_state += 1
    final_state = next_state
    next_state += 1

    print("{src}\t{dest}\t{phone}\t{word}\t{cost}".format(
        src=start_state, dest=shared_state,
        phone='#nonterm_begin', word='#nonterm_begin',
        cost=0.0))

    for nonterminal in nonterminals:
        print("{src}\t{dest}\t{phone}\t{word}\t{cost}".format(
            src=loop_state, dest=shared_state,
            phone=nonterminal, word=nonterminal,
            cost=0.0))
    # this_cost equals log(len(left_context_phones)) but the expression below
    # better captures the meaning.  Applying this cost to arcs keeps the FST
    # stochatic (sum-to-one, like an HMM), so that if we do weight pushing
    # things won't get weird.  In the grammar-FST code when we splice things
    # together we will cancel out this cost, see the function CombineArcs().
    this_cost = -math.log(1.0 / len(left_context_phones))

    for left_context_phone in left_context_phones:
        print("{src}\t{dest}\t{phone}\t{word}\t{cost}".format(
            src=shared_state, dest=loop_state,
            phone=left_context_phone, word='<eps>', cost=this_cost))
    # arc from loop-state to a final-state with #nonterm_end as ilabel and olabel
    print("{src}\t{dest}\t{phone}\t{word}\t{cost}".format(
        src=loop_state, dest=final_state,
        phone='#nonterm_end', word='#nonterm_end', cost=0.0))
    print("{state}\t{final_cost}".format(
        state=final_state, final_cost=0.0))
    return next_state


def write_fst_with_silence(lexicon, sil_prob, sil_phone, sil_disambig, file,
                           nonterminals=None, left_context_phones=None):
    """Writes the text format of L.fst to the standard output.  This version is for
       when --sil-prob != 0.0, meaning there is optional silence
     'lexicon' is a list of 3-tuples (word, pron-prob, prons)
         as returned by read_lexiconp().
     'sil_prob', which is expected to be strictly between 0.. and 1.0, is the
         probability of silence
     'sil_phone' is the silence phone, e.g. "SIL".
     'sil_disambig' is either None, or the silence disambiguation symbol, e.g. "#5".
     'nonterminals', which relates to grammar decoding (see kaldi-asr.org/doc/grammar.html),
        is either None, or the user-defined nonterminal symbols as a list of
        strings, e.g. ['#nonterm:contact_list', ... ].
     'left_context_phones', which also relates to grammar decoding, and must be
        supplied if 'nonterminals' is supplied is either None or a list of
        phones that may appear as left-context, e.g. ['a', 'ah', ... '#nonterm_bos'].
    """

    assert sil_prob > 0.0 and sil_prob < 1.0
    sil_cost = -math.log(sil_prob)
    no_sil_cost = -math.log(1.0 - sil_prob);

    start_state = 0
    loop_state = 1  # words enter and leave from here
    sil_state = 2  # words terminate here when followed by silence; this state
    # has a silence transition to loop_state.
    next_state = 3  # the next un-allocated state, will be incremented as we go.

    print('{src}\t{dest}\t{phone}\t{word}\t{cost}'.format(
        src=start_state, dest=loop_state,
        phone='<eps>', word='<eps>', cost=no_sil_cost), file=file)
    print('{src}\t{dest}\t{phone}\t{word}\t{cost}'.format(
        src=start_state, dest=sil_state,
        phone='<eps>', word='<eps>', cost=sil_cost), file=file)
    if sil_disambig is None:
        print('{src}\t{dest}\t{phone}\t{word}\t{cost}'.format(
            src=sil_state, dest=loop_state,
            phone=sil_phone, word='<eps>', cost=0.0), file=file)
    else:
        sil_disambig_state = next_state
        next_state += 1
        print('{src}\t{dest}\t{phone}\t{word}\t{cost}'.format(
            src=sil_state, dest=sil_disambig_state,
            phone=sil_phone, word='<eps>', cost=0.0), file=file)
        print('{src}\t{dest}\t{phone}\t{word}\t{cost}'.format(
            src=sil_disambig_state, dest=loop_state,
            phone=sil_disambig, word='<eps>', cost=0.0), file=file)

    for (word, pronprob, pron) in lexicon:
        # pron_cost = -math.log(pronprob)
        pron_cost = pronprob
        cur_state = loop_state
        for i in range(len(pron) - 1):
            print("{src}\t{dest}\t{phone}\t{word}\t{cost}".format(
                src=cur_state, dest=next_state,
                phone=pron[i],
                word=(word if i == 0 else '<eps>'),
                cost=(pron_cost if i == 0 else 0.0)), file=file)
            cur_state = next_state
            next_state += 1

        i = len(pron) - 1  # note: i == -1 if pron is empty.
        print("{src}\t{dest}\t{phone}\t{word}\t{cost}".format(
            src=cur_state,
            dest=loop_state,
            phone=(pron[i] if i >= 0 else '<eps>'),
            word=(word if i <= 0 else '<eps>'),
            cost=no_sil_cost + (pron_cost if i <= 0 else 0.0)), file=file)
        print("{src}\t{dest}\t{phone}\t{word}\t{cost}".format(
            src=cur_state,
            dest=sil_state,
            phone=(pron[i] if i >= 0 else '<eps>'),
            word=(word if i <= 0 else '<eps>'),
            cost=sil_cost + (pron_cost if i <= 0 else 0.0)), file=file)

    if nonterminals is not None:
        next_state = write_nonterminal_arcs(
            start_state, loop_state, next_state,
            nonterminals, left_context_phones)

    print("{state}\t{final_cost}".format(
        state=loop_state,
        final_cost=0.0), file=file)


def prepare_lexicon(lexicon, silence_phones, nonsilence_phones, optional_silence, oov):
    """Converts a lexicon as a list of transcriptions into an FST.
    
        'lexicon' is a list of 3-tuples (word, pron-prob, prons)
        'silence_phones' list of phonemes representing silence, noise, etc.
        'nonsilence_phones' list of phonemes being a realization of speech
        'optional_silnce' phoneme representing inter-word silence
        'oov' token for out-of-vocabulary/unknown words (eg. <unk>)
        
        Returns:
            input_syms,output_syms,FST - symbols and FST of the lexicon
    """
    for w, p, trans in lexicon:
        for ph in trans:
            assert ph in nonsilence_phones or ph in silence_phones, f'ERROR: {ph} is not a proper phoneme!'
    lexicon = copy.deepcopy(lexicon)
    for w, p, trans in lexicon:
        if len(trans) == 1:
            trans[0] += '_S'
        else:
            trans[0] += '_B'
            if len(trans) > 2:
                for i in range(1, len(trans) - 1):
                    trans[i] += '_I'
            trans[-1] += '_E'

    # add_disambig
    first_sym = 1  # 0 is reserved for wdisambig
    max_disambig = first_sym - 1
    reserved_empty = set()
    last_sym = {}

    count = {}  # number of identical transcripstions
    for w, p, trans in lexicon:
        x = ' '.join(trans)
        if x not in count:
            count[x] = 1
        else:
            count[x] += 1

    prefix = set()  # set of all possible prefixes (does not include full transcriptions)
    for w, p, trans in lexicon:
        t = trans.copy()
        while len(t) > 0:
            t.pop()
            prefix.add(' '.join(t))

    for w, p, trans in lexicon:
        x = ' '.join(trans)
        if x not in prefix and count[x] == 1:
            pass
        else:
            if len(x) == 0:
                max_disambig += 1
                reserved_empty.add(max_disambig)
                trans.append(f'#{max_disambig}')
            else:
                if x not in last_sym:
                    curr_sym = first_sym
                else:
                    curr_sym = last_sym[x] + 1
                while curr_sym in reserved_empty:
                    curr_sym += 1
                if curr_sym > max_disambig:
                    max_disambig = curr_sym
                last_sym[x] = curr_sym
                trans.append(f'#{curr_sym}')

    max_disambig += 1
    sil_disambig = max_disambig

    counter = 0
    phone_map = {}
    phone_map['<eps>'] = counter
    counter += 1
    for ph in silence_phones:
        phone_map[ph] = counter
        counter += 1
        phone_map[ph + '_B'] = counter
        counter += 1
        phone_map[ph + '_E'] = counter
        counter += 1
        phone_map[ph + '_S'] = counter
        counter += 1
        phone_map[ph + '_I'] = counter
        counter += 1
    for ph in nonsilence_phones:
        phone_map[ph + '_B'] = counter
        counter += 1
        phone_map[ph + '_E'] = counter
        counter += 1
        phone_map[ph + '_S'] = counter
        counter += 1
        phone_map[ph + '_I'] = counter
        counter += 1
    for i in range(max_disambig + 1):
        phone_map[f'#{i}'] = counter
        counter += 1

    isyms = fst.SymbolTable()
    with open('phones.txt', 'w') as f:
        for p, i in phone_map.items():
            f.write(f'{p} {i}\n')
            isyms.add_symbol(p, i)

    with open('disambig.int', 'w') as f:
        for i in range(max_disambig + 1):
            d = phone_map[f'#{i}']
            f.write(f'{d}\n')

    counter = 0
    words_map = {}
    words_map['<eps>'] = counter
    counter += 1
    wordlist = set()
    for w, p, t in lexicon:
        wordlist.add(w)
    for w in sorted(wordlist):
        words_map[w] = counter
        counter += 1
    words_map['#0'] = counter
    counter += 1
    words_map['<s>'] = counter
    counter += 1
    words_map['</s>'] = counter

    osyms = fst.SymbolTable()
    with open('words.txt', 'w') as f:
        for w, i in words_map.items():
            f.write(f'{w} {i}\n')
            osyms.add_symbol(w, i)

    with open('word_boundary.int', 'w') as f:
        cnt = 1
        for i in range(len(silence_phones)):
            for b in ['nonword', 'begin', 'end', 'singleton', 'internal']:
                f.write(f'{cnt} {b}\n')
                cnt += 1
        for i in range(len(nonsilence_phones)):
            for b in ['begin', 'end', 'singleton', 'internal']:
                f.write(f'{cnt} {b}\n')
                cnt += 1

    compiler = fst.Compiler(isymbols=isyms, osymbols=osyms)
    write_fst_with_silence(lexicon, 0.5, optional_silence, None, compiler, nonterminals=None, left_context_phones=None)

    L = compiler.compile()
    L.arcsort(sort_type='olabel')

    return isyms, osyms, L


nonsilence_phones = sorted(['I', 'S', 'Z', 'a', 'b', 'd', 'dZ', 'dz', 'dzi', 'e', 'en', 'f', 'g', 'i', 'j', 'k', 'l',
                            'm', 'n', 'ni', 'o', 'on', 'p', 'r', 's', 'si', 't', 'tS', 'ts', 'tsi', 'u', 'v', 'w', 'x',
                            'z', 'zi'])
silence_phones = sorted(['sil', 'spn'])
optional_silence = 'sil'
oov = '<unk>'


def words_to_lexicon(wordlist, model=Path('phonetisaurus/model.fst')):
    """Converts a list of words to a lexicon FST.
    :param wordlist: list of word strings
    :param model: Phonetisaurus G2P model (default: phonetisaurus/model.fst)
    :return: (psyms,wsyms,L.fst)
    """

    with NamedTemporaryFile(mode='w', prefix='wlist', delete=False) as ntf:
        tmp_wlist = Path(ntf.name)
        for w in wordlist:
            ntf.write(f'{w}\n')
    with NamedTemporaryFile(prefix='lex', delete=False) as ntf:
        tmp_lex = Path(ntf.name)
        cmd = f'phonetisaurus-g2pfst --model={model} --pmass=0.8 --wordlist={tmp_wlist}'
        run(shlex.split(cmd), stdout=ntf)

    lexicon = []
    with open(tmp_lex) as f:
        for l in f:
            tok = l.strip().split('\t')
            lexicon.append((tok[0], float(tok[1]), tok[2].split()))

    tmp_wlist.unlink()
    tmp_lex.unlink()

    return prepare_lexicon(lexicon, silence_phones, nonsilence_phones, optional_silence, oov)


# unit test
if __name__ == '__main__':
    psyms, wsyms, L = words_to_lexicon(['ala', 'ma', 'kota'])

    L.set_input_symbols(psyms)
    L.set_output_symbols(wsyms)
    L.write('L.fst')
