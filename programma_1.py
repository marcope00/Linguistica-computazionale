# PROGRAMMA 1 - MARCO PETRUCCI [598657]

 # Confrontate i due testi sulla base delle seguenti informazioni statistiche:
    # • il numero di frasi e di token;
    # • la lunghezza media delle frasi in termini di token e dei token (escludendo la punteggiatura) in termini di caratteri;
    # • il numero di hapax sui primi 1000 token;
    # • la grandezza del vocabolario e la ricchezza lessicale calcolata attraverso la Type Token Ratio (TTR), in entrambi i casi calcolati all'aumentare del corpus per porzioni incrementali di 500 token;
 # • distribuzione in termini di percentuale dell'insieme delle parole piene (Aggettivi, Sostantivi, Verbi, Avverbi) e delle parole funzionali (Articoli, Preposizioni, Congiunzioni, Pronomi).

import sys
import nltk

LISTA_PUNTEGGIATURA = ['.', ',', ';', ':', '?', '!'] # Costante che contiene tutta la punteggiatura
PAROLE_PIENE_POS = ['JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'NNPS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'RB', 'RBR', 'RBS'] # Costante che contiene tutte le POS delle parole piene
PAROLE_FUNZIONALI_POS = ['CC', 'CD', 'DT', 'EX', 'IN', 'LS', 'PDT', 'POS', 'PRP', 'PRP$', 'TO', 'WDT', 'WP', 'WP$', 'WRB'] # Costante che contiene tutte le POS delle parole funzionali


def analizza_corpus(testo):
    sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    riga = sent_tokenizer.tokenize(testo) # Variabile che contiene tutte le frasi del corpus
    lista_tok, num_frasi, num_tok, len_char, num_token_filtr, num_hapax, num_p_piene, num_p_funzionali = [], 0, 0, 0, 0, 0, 0, 0

   # Scorre ogni frase contenuta nel corpus e aumenta di uno il numero delle frasi ogni volta che il ciclo for viene scorso
    for frase in riga:
        num_frasi += 1
        tokens = nltk.word_tokenize(frase)
        tokens_e_POS = nltk.pos_tag(tokens)
        lista_tok += tokens

       # Filtra i token togliendo la punteggiatura
        for token in tokens:
            num_tok += 1 # Aumenta il numero di token
            if token not in LISTA_PUNTEGGIATURA:
                len_char += len(token)
                num_token_filtr += 1 # Aumenta il numero di token filtrati

       # Individua i token con Part of Speech tagging uguale a parole piene o parole funzionali
        for token_e_POS in tokens_e_POS:
            if token_e_POS[1] in PAROLE_PIENE_POS:
                num_p_piene += 1
            if token_e_POS[1] in PAROLE_FUNZIONALI_POS:
                num_p_funzionali += 1

   # Calcola la media della lunghezza delle frasi e dei token
    len_avg_f = num_tok/num_frasi
    len_avg_t = len_char/num_token_filtr

   # Calcola il numero di hapax dei soli primi 1000 tokens del corpus
    lista_tok1000 = lista_tok[0:1000]
    freq_token1000 = nltk.FreqDist(lista_tok1000).most_common() # Calcola la distribuzione dei primi 1000 token
    for freq in freq_token1000:
        if freq[1] == 1: # Se il token ha frequenza 1 aumenta il numero degli hapax
            num_hapax += 1

   # Calcola la distribuzione in termini di percentuale dell'insieme delle parole piene e delle parole funzionali
    dist_p_piene = (num_p_piene/num_tok)*100
    dist_p_funzionali = (num_p_funzionali/num_tok)*100

    return num_frasi, lista_tok, num_tok, len_avg_f, len_avg_t, num_hapax, dist_p_piene, dist_p_funzionali


def analizza_corpus500(file, lista_tok, num_tok):
    print('\n• File analizzato:', '"' + file + '"')
   # Andamento della crescita lessicale del file all'aumentare del testo (500 token per volta)
    for i in range(0, num_tok, 500):
        lista_tok500 = lista_tok[0:i+500] # Seleziona i primi 500 tokens del corpus + quelli già scorsi
        vocabolario500 = list(set(lista_tok500)) # Restituisce gli elementi diversi presenti nella lista dei 500 tokens scorsi 

       # Numero delle parole tipo e Type Token Ratio dei primi i+500 tokens del corpus
        numero_parole_tipo500 = len(vocabolario500)
        ttr500 = len(vocabolario500)/num_tok

       # Aggiunge un contatore dei token scorsi: nel caso vengano scorsi tutti i token cambia il valore
        limite = i + 500
        if limite > num_tok:
            limite = "aver scorso tutti i"

        print("  Numero di parole tipo dopo", limite, "tokens:", numero_parole_tipo500)
        print("  La Type Token Ratio dopo", limite, "tokens:", ttr500, "\n")


def main(file1, file2):
    fileInput1 = open(file1, mode="r", encoding="utf-8")
    fileInput2 = open(file2, mode="r", encoding="utf-8")
    testo1 = fileInput1.read()
    testo2 = fileInput2.read()

    num_frasi1, lista_tok1, num_tok1, len_avg_f1, len_avg_t1, num_hapax1, dist_p_piene1, dist_p_funzionali1 = analizza_corpus(testo1)
    num_frasi2, lista_tok2, num_tok2, len_avg_f2, len_avg_t2, num_hapax2, dist_p_piene2, dist_p_funzionali2 = analizza_corpus(testo2)

# Stampa i risultati

   # NUMERO DI FRASI E DI TOKEN
    print("• Il file", '"'+file1+'" ha', num_frasi1, "frasi e", num_tok1, "tokens.")
    print("  Il file", '"'+file2+'" ha', num_frasi2, "frasi e", num_tok2, "tokens.\n")

   # LUNGHEZZA MEDIA DELLE FRASI
    print("• La lunghezza media delle frasi del file", '"'+file1+'" è di', len_avg_f1)
    print("  La lunghezza media delle frasi del file", '"'+file2+'" è di', len_avg_f2)
    print("  La lunghezza media dei tokens del file", '"'+file1+'" è di', len_avg_t1, "(escludendo la punteggiatura).")
    print("  La lunghezza media dei tokens del file", '"'+file2+'" è di', len_avg_t2, "(escludendo la punteggiatura).\n")

   # NUMERO HAPAX DEI PRIMI 1000 TOKENS
    print("• Gli hapax dei primi 1000 tokens del file", '"'+file1+'" sono', num_hapax1)
    print("  Gli hapax dei primi 1000 tokens del file", '"'+file2+'" sono', num_hapax2)
    print("_________________________________________________")

   # CRESCITA LESSICALE DEL CORPUS OGNI 500 TOKENS
    analizza_corpus500(file1, lista_tok1, num_tok1)
    print("-------------------------------------------------")
    analizza_corpus500(file2, lista_tok2, num_tok2)
    print("_________________________________________________\n")

   # DISTRIBUZIONE DELLE PAROLE PIENE E FUNZIONALI
    print("• Le parole piene del file", '"'+file1+'" costituiscono il', str(dist_p_piene1) + "% del corpus.")
    print("  Le parole funzionali del file", '"'+file1+'" costituiscono il', str(dist_p_funzionali1) + "% del corpus.")
    print("  Le parole piene del file", '"'+file2+'" costituiscono il', str(dist_p_piene2) + "% del corpus.")
    print("  Le parole funzionali del file", '"'+file2+'" costituiscono il', str(dist_p_funzionali2) + "% del corpus.")


main(sys.argv[1], sys.argv[2])