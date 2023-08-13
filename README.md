# Intelligibility of Slavic Languages for Russian Native Speakers

This page contains supplimentary materials for the article "Intelligibility of Slavic Languages for Russian Native Speakers" to be sent in JQL. 
It consists of [data](https://github.com/slavicintelligibility2023/russian_intelligibility/tree/main/data) folder 
which contains result of test of Russian native speakers and 
[Jupyter notebook](https://github.com/slavicintelligibility2023/russian_intelligibility/blob/main/intelligibility_figures.py) 
which generates figures for the article.

### The main idea
It has long been recognized that speakers of related languages can have successful communication speaking their native languages. Various terms have been used to refer to such kind of communication like semicommunication, receptive multilingualism, semibilingualism, inherent intelligibility, non-convergent discourse, asymmetric/bilingual discourse.

One of the most popular methods is a cloze test, a method which has been devised for language proficiency testing and successfully applied in psychology, machine translation testing and other spheres. 

In this paper we aimed to study the intelligibility of Slavic languages to native speakers of Russian. To do it we conducted cloze test experiments over pieces of text in Russian translated into eight Slavic languages of three different families: Ukrainian and Belarusian (East Slavic), Polish, Czech, and Slovak (West Slavic), and Serbian, Slovene, and Bulgarian (South Slavic).

In our experiment participants were given a text in Russian with omitted words and a parallel text in a Slavic language (unknown to them) to use as a clue. The technique was different from the cloze test experiments in previous papers, where participants had a list of response alternatives translated into their native language with which to complete a gapped text in a foreign language. In our case, there were no restrictions on the answers other than the ones imposed by the context in Russian and the clue from a corresponding word in another Slavic language (provided the participant identified it correctly in the sentence). This way of arranging the experiment does not limit participants in their choice of words and allows gathering objective quantitative information on a text intelligibility. 

If a word was omitted in the text once, it was to be omitted in all other occurrences in this test (so that it would not be possible to guess it in case its equivalent is repeated in the parallel text). Words (totalling 186 lemmas, 232 word occurrences) were omitted in the Russian part of a test independently from translations.

Of the nine languages used in the study, five (including Russian) use Cyrillic alphabet and four use Latin alphabet. To avoid problems with the extended Latin script, we transliterated such parallel texts into Cyrillic using simple rules of substitution. For the sake of consistency all parallel texts written in Cyrillic script were transliterated into standard Latin. Thus, subjects saw each parallel text written both in Latin and Cyrillic scripts aligned by sentences.

It is possible to restore words in a gapped text using the text redundancy on the basis of limitations imposed by the context itself. To identify this influence each of the six test pieces was given to a control group of Russian native speakers to do without a parallel text in another Slavic language.

All participants did tests on a web site without any observation of their behaviour. At the beginning each participant was asked to select their level of education, to indicate if they knew any other Slavic language, and if their education included any language studies. All Slavic languages named by the participant were excluded from the list from which the parallel language for the test was randomly selected. It was also possible to skip these three questions, then the subject was given a control test (a gapped text in Russian without a parallel text).

In the experiment we used three pieces from M. Bulgakov’s Master and Margaret (originally written in Russian), three pieces of H. Sienkiewicz’s Quo Vadis? (originally written in Polish), and their modern translations into the other languages in the study.


### Structure of the repository.

- [data](https://github.com/slavicintelligibility2023/russian_intelligibility/tree/main/data) folder contains the smapled data. It contains README.md which describes the stucture of the data.

- [Jupyter notebook](https://github.com/slavicintelligibility2023/russian_intelligibility/blob/main/intelligibility_figures.py) generates figures for the article. Contains just interactive interface.

- [img_res](https://github.com/slavicintelligibility2023/russian_intelligibility/tree/main/img_res) folder which contains resulting figures.

- [intelligibility_figures.py](https://github.com/slavicintelligibility2023/russian_intelligibility/blob/main/intelligibility_figures.py) Pyhton file with main functions which are really generating figures.

