# Description of Data Structure

- **user_data.csv** - participants' data. Fields description:
    - *id_user* - identifier of a participant.
    - *test_id* - identifier of a test passed by the participant.
    - *parallel_lang* - a language used for a parallel text. *No Parallel Text* denotes the control group, i.e. those participants who restored words without the parallel text. 
    - *age* - age group of the participant. *sch* denotes school children, bak - bachelor students, mag - master student, fin - finished his or her education.
    - *speciality* - the participant's education included any language studies (translation, linguistics, philology): yes - includes, no - don't includes or was not selected.
    - *known_langs* - if the participant knows any other Slavic language. Five and more languages denote situation when the participant declined any parallel text and volonteered to be in the control group.
    - *ans_cnt* - number of non-empty answers for this participant.
    
    
- **user_answers.csv** - participants' answers. Fields description:
    - *id_user* - identifier of a participant; equals to the same field in the *user_data* table.
    - *test_id* - identifier of a test passed by the participant.
    - *parallel_lang* - a language used for a parallel text; equals to the same field in the *user_data* table.
    - *answer_no* - position of a omitted word in the test.
    - *answer* - participant's answer.
    - *correctness* - asessors' evaluation of the answer. incorr denotes an incorrect answer, pcorr - partially correct anser, corr - correct answer.
    
    
- **texts.csv** - text phragments used as tests. The words, marked by the underscore denote omitted words in parallel texts. All texts are aligned by sentences, i.e. have the same identifier for different languages. Fields description:
    - *id_set* - identifier of a text.
    - *lang_name* - language of the text.
    - *sent_pos* - position of a sentence in the text.
    - *sent* - text of the sentence.
