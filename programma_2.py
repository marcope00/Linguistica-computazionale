# PROGRAMMA 2 - MARCO PETRUCCI [598657]

 # Per ognuno dei due corpora estraete le seguenti informazioni:
 # • estraete ed ordinate in ordine di frequenza decrescente, indicando anche la relativa frequenza:
    # ◦ le 10 PoS (Part-of-Speech) più frequenti;
    # ◦ i 10 bigrammi di PoS più frequenti;
    # ◦ i 10 trigrammi di PoS più frequenti;
    # ◦ i 20 Aggettivi e i 20 Avverbi più frequenti;
 
 # • estraete ed ordinate i 20 bigrammi composti da Aggettivo e Sostantivo e dove ogni token ha una frequenza maggiore di 3:
    # ◦ con frequenza massima, indicando anche la relativa frequenza;
    # ◦ con probabilità condizionata massima, indicando anche la relativa probabilità;
    # ◦ con forza associativa (calcolata in termini di Local Mutual Information) massima, indicando anche la relativa forza associativa;
 
 # • estraete le frasi con almeno 6 token e più corta di 25 token, dove ogni singolo token occorre almeno due volte nel corpus di riferimento:
    # ◦ con la media della distribuzione di frequenza dei token più alta, in un caso, e più bassa nell'altro, riportando anche la distribuzione media di frequenza. La distribuzione media di frequenza deve essere calcolata tenendo in considerazione la frequenza di tutti i token presenti nella frase (calcolando la frequenza nel corpus dal quale la frase è stata estratta) e dividendo la somma delle frequenze per il numero di token della frase stessa;
    # ◦ con probabilità più alta, dove la probabilità deve essere calcolata attraverso un modello di Markov di ordine 2. Il modello deve usare le statistiche estratte dal corpus che contiene le frasi;
 
 # • dopo aver individuato e classificato le Entità Nominate (NE) presenti nel testo, estraete:
    # ◦ i 15 nomi propri di persona più frequenti (tipi), ordinati per frequenza.

import sys
import nltk
from nltk import ngrams
import math


def analizza_corpus(testo):
    sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    riga = sent_tokenizer.tokenize(testo)
    lista_tok_e_POS, lista_tok, lista_POS = [], [], []

   # Scorre ogni frase contenuta nel corpus e aumenta di uno il numero delle frasi ogni volta che il ciclo for viene scorso
    for frase in riga:
        tokensf = nltk.word_tokenize(frase)
        tokensf_e_POS = nltk.pos_tag(tokensf)

        for token_e_POS in tokensf_e_POS:
            lista_tok_e_POS.append(token_e_POS)
            lista_tok.append(token_e_POS[0])
            lista_POS.append(token_e_POS[1])

   # Calcola la frequenza delle prime 10 PoS più frequenti
    distr10_POS = nltk.FreqDist(lista_POS).most_common(10)

   # Estrai informazioni utili successivamente dal testo 
    corpus = len(lista_tok)
    bigrammi = list(ngrams(lista_tok, 2))
    distribuzione_freq_token = nltk.FreqDist(lista_tok)
    trigrammi = list(ngrams(lista_tok, 3))

    return lista_tok_e_POS, lista_POS, distr10_POS, corpus, riga, distribuzione_freq_token, bigrammi, trigrammi


def EstraiFrequenzeNgrammi(lista_POS, num_ngramma):
   # Disponi i token in n-grammi a seconda del numero della variabile num_ngramma
    ngrammiPOS = list(ngrams(lista_POS, num_ngramma))

   # Stampa i primi 10 POS di n-grammi con frequenza maggiore
    distribuzione_first10 = nltk.FreqDist(ngrammiPOS).most_common(10)
    for elem in distribuzione_first10:
        print(elem[0], '\tfrequenza:', elem[1])


def aggettivi_avverbi20(lista_tok_e_POS):
    AGG, AVV, lista_AGG, lista_AVV, aggettivi, avverbi = ['JJ', 'JJR', 'JJS'], ['RB', 'RBR', 'RBS'], [], [], '', ''
    distribuzione = nltk.FreqDist(lista_tok_e_POS).most_common()

   # Se il token è un aggettivo, stampa i 20 con la frequenza maggiore nel corpus
    for elem in distribuzione:
        if elem[0][1] in AGG:
            if len(lista_AGG) != 19:
                lista_AGG.append(elem[0][0])
                aggettivi += elem[0][0] + ', ' # Aggiunge una virgola fra un aggettivo e l'altro
            else:
                aggettivi += elem[0][0] + '.' # Aggiunge un punto dopo che sono stati stampati tutti e 20 gli aggettivi
                break

   # Se il token è un avverbio, stampa i 20 con la frequenza maggiore nel corpus
    for elem in distribuzione:
        if elem[0][1] in AVV:
            if len(lista_AVV) != 19:
                lista_AVV.append(elem[0][0])
                avverbi += elem[0][0] + ', ' # Aggiunge una virgola fra un avverbio e l'altro
            else:
                avverbi += elem[0][0] + '.' # Aggiunge un punto dopo che sono stati stampati tutti e 20 gli avverbi
                break

    return aggettivi, avverbi


def ordina(dict):
   # Riordina il vocabolario in senso decrescente 
    return sorted(dict.items(), key=lambda x: x[1], reverse=True)


def big_aggettivo_sostantivo20(corpus, lista_tok_e_POS, distribuzione_freq_token):
    AGG, SOST, lista_AGG_SOST, lista_freq_ass, lista_prob_cond, lista_forza_assoc = ['JJ', 'JJR', 'JJS'], ['NN', 'NNS', 'NNP', 'NNPS'], [], {}, {}, {}
    bigrammi_tok_POS = list(ngrams(lista_tok_e_POS, 2))

   # Verifica che entrambi i token del bigramma hanno frequenza maggiore di 3 e che il primo sia un aggettivo, il secondo un sostantivo
    for elem in bigrammi_tok_POS:
        freq_tok1 = distribuzione_freq_token[elem[0][0]]
        freq_tok2 = distribuzione_freq_token[elem[1][0]]
        if freq_tok1 > 3 and freq_tok2 > 3 and elem[0][1] in AGG and elem[1][1] in SOST:
            big = (elem[0][0], elem[1][0])
            lista_AGG_SOST.append(big)

   # Calcola la frequenza assoluta, la probabilità condizionata dei primi 20 bigrammi aggettivo-sostantivo
    distribuzione_freq_big = nltk.FreqDist(lista_AGG_SOST)
    for big in lista_AGG_SOST:
        freq_agg = distribuzione_freq_token[big[0]]
        freq_sost = distribuzione_freq_token[big[1]]

       # Calcola i valori di frequenza assoluta del bigramma
        frequenza_big = distribuzione_freq_big[big]
        lista_freq_ass[big] = frequenza_big

       # Calcola i valori di probabilità condizionata del bigramma dalla frequenza del bigramma diviso la frequenza del secondo token del bigramma
        prob_cond = frequenza_big/freq_sost
        lista_prob_cond[big] = prob_cond

       # Calcola la LMI: f(a,b)*log2(P(a,b)/(f(a)/C*f(b)/C))
        p_agg = freq_agg/corpus
        p_sost = freq_sost/corpus

        prob_congiunta = prob_cond * p_agg
        L_Mutual_Inf = frequenza_big * math.log(prob_congiunta / (p_agg * p_sost), 2)
        lista_forza_assoc[big] = L_Mutual_Inf

    # Ordina in senso decrescente i valori
    freq_ass_ordered = list(ordina(lista_freq_ass))
    prob_cond_ordered = list(ordina(lista_prob_cond))
    forza_assoc_ordered = list(ordina(lista_forza_assoc))

    return freq_ass_ordered, prob_cond_ordered, forza_assoc_ordered


def estrai_frasi(riga, corpus, distribuzione_freq_token, bigrammi, trigrammi):
    frase_ALTA_distr, mediaMAX, frase_BASSA_distr, mediaMIN, frase_ALTA_prob, markov2MAX = '' , -math.inf, '', math.inf, '', -math.inf

   # Scorri ogni frase del corpus 
    for frase in riga:
        occorrenza, somma, markov2 = True, 0, 1.0
        tokensf = nltk.word_tokenize(frase)

       # Se il numero di token della frase è almeno di 6 e meno di 25, e ogni token ha una frequenza di almeno 2, esegui le operazioni
        if len(tokensf) >= 6 and len(tokensf) < 25:
            for token in tokensf:
                freq_tok = distribuzione_freq_token[token]
                if occorrenza == True and freq_tok >= 2:
                    somma += freq_tok
                else:
                    occorrenza = False

           # Se tutto rispetta i canoni, trova le frasi con lunghezza media più alta e più bassa 
            if occorrenza == True:
                media = somma/len(tokensf)
                if media > mediaMAX:
                    mediaMAX = media
                    frase_ALTA_distr = frase
                if media < mediaMIN:
                    mediaMIN = media
                    frase_BASSA_distr = frase
                
               # Calcola la probabilità di ogni frase mediante il modello di Markov di ordine 2: P(w1, w2,...,wn) = P(w1)*P(w2|w1)*P(w3|w1,w2)*...*P(wn|wn-2,wn-1)
                bigrammiFrase = list(ngrams(tokensf, 2))
                distribuzione_freq_big = nltk.FreqDist(bigrammi)
                freq_primo_tok = distribuzione_freq_token[bigrammiFrase[0][0]]
                
                P_w1 = freq_primo_tok/corpus
                freq_primo_big = distribuzione_freq_big[bigrammiFrase[0][0], bigrammiFrase[0][1]]
                prob_cond_big = freq_primo_big/freq_primo_tok
                markov2 = P_w1 * prob_cond_big

                trigrammi_f = list(ngrams(tokensf, 3))
                distribuzione_freq_trig = nltk.FreqDist(trigrammi)
                for elem in trigrammi_f:
                    frequenza_trig = distribuzione_freq_trig[elem[0], elem[1], elem[2]]
                    frequenza_big = distribuzione_freq_big[elem[0], elem[1]]
                    prob_cond_trig = frequenza_trig/frequenza_big
                    markov2 *= prob_cond_trig

               # Trova la frase con probabilità più alta
                if markov2 > markov2MAX:
                    markov2MAX = markov2
                    frase_ALTA_prob = frase
    
    return frase_ALTA_distr, mediaMAX, frase_BASSA_distr, mediaMIN, frase_ALTA_prob, markov2MAX


def estrai_nomi_persona(lista_tok_e_POS):
    personeTOT, lista15_persone, count = [], '', 0

   # Esegue il Parser Context-Free dei tokens del corpus
    analisi = nltk.ne_chunk(lista_tok_e_POS)
    for nodo in analisi:
        Ne = ''

       # Se il token è una persona, lo aggiunge alla lista di tutti i nomi di persona  
        if hasattr(nodo, 'label') and nodo.label() == 'PERSON':
            for persona in nodo.leaves():
                Ne += '' + persona[0]
            personeTOT.append(Ne)

   # Calcola la frequenza di ogni nome di persona 
    dist_persone = nltk.FreqDist(personeTOT).most_common(15)
    for elem in dist_persone:
        if count != 14:
            lista15_persone += elem[0] + ', ' # Aggiunge una virgola fra una persona e l'altra
        else:
            lista15_persone += elem[0] + '.' # Aggiunge un punto dopo che sono stati stampate tutte e 20 le persone
        count += 1

    return lista15_persone


def main(file1, file2):
    fileInput1 = open(file1, mode="r", encoding="utf-8")
    fileInput2 = open(file2, mode="r", encoding="utf-8")
    testo1 = fileInput1.read()
    testo2 = fileInput2.read()

# Stampa i risultati

   # PRIME 10 POS
    print("• LISTA DELLE 10 POS PIU' FREQUENTI")
    print(' - del file', '"' + file1 + '":')
    lista_tok_e_POS1, lista_POS1, distr10_POS1, corpus1, riga1, distribuzione_freq_token1, bigrammi1, trigrammi1 = analizza_corpus(testo1)
    for elem in distr10_POS1:
        print("POS:", elem[0], '\tfrequenza:', elem[1])

    print('\n - del file', '"' + file2 + '":')
    lista_tok_e_POS2, lista_POS2, distr10_POS2, corpus2, riga2, distribuzione_freq_token2, bigrammi2, trigrammi2 = analizza_corpus(testo2)
    for elem in distr10_POS2:
        print("POS:", elem[0], '\tfrequenza:', elem[1])

    print("\n_________________________________________________")

   # PRIMI 10 BIGRAMMI
    print('\n• PRIMI 10 BIGRAMMI DI POS IN ORDINE DI FREQUENZA')
    print(' - del file', '"' + file1 + '":')
    EstraiFrequenzeNgrammi(lista_POS1, 2)

    print('\n - del file', '"' + file2 + '":')
    EstraiFrequenzeNgrammi(lista_POS2, 2)

    print("\n_________________________________________________")

   # PRIMI 10 TRIGRAMMI
    print('\n• PRIMI 10 TRIGRAMMI DI POS IN ORDINE DI FREQUENZA')
    print(' - del file', '"' + file1 + '":')
    EstraiFrequenzeNgrammi(lista_POS1, 3)

    print('\n - del file', '"' + file2 + '":')
    EstraiFrequenzeNgrammi(lista_POS2, 3)

    print("\n_________________________________________________")

   # PRIMI 20 AGGETTIVI E AVVERBI
    aggettivi1, avverbi1 = aggettivi_avverbi20(lista_tok_e_POS1)
    aggettivi2, avverbi2 = aggettivi_avverbi20(lista_tok_e_POS2)
    print("\n• I 20 AGGETTIVI PIU' FREQUENTI")
    print(' - del file', '"' + file1 + '":\n', aggettivi1)
    print('\n - del file', '"' + file2 + '":\n', aggettivi2)

    print("\n• I 20 AVVERBI PIU' FREQUENTI")
    print(' - del file', '"' + file1 + '":\n', avverbi1)
    print('\n - del file', '"' + file2 + '":\n', avverbi2)

    print("\n_________________________________________________")

   # PRIMI 10 BIGRAMMI AGGETTIVO-SOSTANTIVO
    freq_ass_ordered1, prob_cond_ordered1, forza_assoc1 = big_aggettivo_sostantivo20(corpus1, lista_tok_e_POS1, distribuzione_freq_token1)
    freq_ass_ordered2, prob_cond_ordered2, forza_assoc2 = big_aggettivo_sostantivo20(corpus2, lista_tok_e_POS2, distribuzione_freq_token2)
    
    print('\n• PRIMI 20 BIGRAMMI AGGETTIVO-SOSTANTIVO (ENTRAMBI CON FREQUENZA > 3)')
    print('| CON FREQUENZA MASSIMA')
    print(' - del file', '"' + file1 + '":')
    for tupla in freq_ass_ordered1[0:20]:
        print(tupla[0][0], tupla[0][1], "\tfrequenza assoluta:", tupla[1])
    print('\n - del file', '"' + file2 + '":')
    for tupla in freq_ass_ordered2[0:20]:
        print(tupla[0][0], tupla[0][1], "\tfrequenza assoluta:", tupla[1])

    print("\n| CON PROBABILITA' CONDIZIONATA MASSIMA")
    print(" - del file", '"' + file1 + '":')
    for tupla in prob_cond_ordered1[0:20]:
        print(tupla[0][0], tupla[0][1], "\tprobabilità condizionata:", tupla[1])
    print("\n - del file", '"' + file2 + '":')
    for tupla in prob_cond_ordered2[0:20]:
        print(tupla[0][0], tupla[0][1], "\tprobabilità condizionata:", tupla[1])
    
    print('\n| CON FORZA ASSOCIATIVA MASSIMA')
    print(' - del file', '"' + file1 + '":')
    for tupla in forza_assoc1[0:20]:
        print(tupla[0][0], tupla[0][1], "\tLMI:", tupla[1])
    print('\n - del file', '"' + file2 + '":')
    for tupla in forza_assoc2[0:20]:
        print(tupla[0][0], tupla[0][1], "\tLMI:", tupla[1])

    print("\n_________________________________________________")

   #FRASI
    print("\n• FRASE CON ALMENO 6 E MENO DI 25 TOKEN (DOVE OGNI TOKEN HA UNA FREQUENZA > 2)")
    frase_ALTA_distr1, mediaMAX1, frase_BASSA_distr1, mediaMIN1, frase_ALTA_prob1, markov2MAX1 = estrai_frasi(riga1, corpus1, distribuzione_freq_token1, bigrammi1, trigrammi1)
    frase_ALTA_distr2, mediaMAX2, frase_BASSA_distr2, mediaMIN2, frase_ALTA_prob2, markov2MAX2 = estrai_frasi(riga2, corpus2, distribuzione_freq_token2, bigrammi2, trigrammi2)

    print("| CON LA MEDIA DELLA DISTRIBUZIONE DI FREQUENZA DEI TOKEN PIU' ALTA")
    print(" - del file", '"' + file1 + '":\n', frase_ALTA_distr1, "\tmedia di distribuzione:", mediaMAX1)
    print("\n - del file", '"' + file2 + '":\n', frase_ALTA_distr2, "\tmedia di distribuzione:", mediaMAX2)

    print("\n| CON LA MEDIA DELLA DISTRIBUZIONE DI FREQUENZA DEI TOKEN PIU' BASSA")
    print(" - del file", '"' + file1 + '":\n', frase_BASSA_distr1, "\tmedia di distribuzione:", mediaMIN1)
    print("\n - del file", '"' + file2 + '":\n', frase_BASSA_distr2, "\tmedia di distribuzione:", mediaMIN2)

    print("\n| CON LA PROBABILITA' PIU' ALTA, CALCOLATA ATTRAVERSO IL MODELLO DI MARKOV DI ORDINE 2")
    print(" - del file", '"' + file1 + '":\n', frase_ALTA_prob1, "\tprobabilità:", markov2MAX1)
    print("\n - del file", '"' + file2 + '":\n', frase_ALTA_prob2, "\tprobabilità:", markov2MAX2)

    print("\n_________________________________________________")

   # PRIMI 15 NOMI PROPRI DI PERSONA
    print("\n• I PRIMI 15 NOMI PROPRI DI PERSONA PIU' FREQUENTI")
    persone1 = estrai_nomi_persona(lista_tok_e_POS1)
    print(" - del file", '"' + file1 + '":\n', persone1)

    persone2 = estrai_nomi_persona(lista_tok_e_POS2)
    print("\n - del file", '"' + file2 + '":\n', persone2)

    
main(sys.argv[1], sys.argv[2])