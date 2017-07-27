import bz2
import cPickle
import os

from snorkel.contrib.babble import Explanation

def get_user_lists():
    ctd_pickle_path = os.path.join(os.environ['SNORKELHOME'], 
                                   'tutorials/cdr/data/ctd.pkl.bz2') 
    print('Loading canonical ID ontologies...')
    with bz2.BZ2File(ctd_pickle_path, 'rb') as ctd_f:
        ctd_unspecified, ctd_therapy, ctd_marker = cPickle.load(ctd_f)
    print('Finished loading canonical ID ontologies.')
    user_lists = {
        'uncertain': ['combin', 'possible', 'unlikely'],
        'causal': ['causes', 'caused', 'induce', 'induces', 'induced', 'associated with'],
        'treat': ['treat', 'effective', 'prevent', 'resistant', 'slow', 'promise', 'therap'],
        'procedure': ['inject', 'administrat'],
        'patient': ['in a patient with', 'in patients with'],
        'weak': ['none', 'although', 'was carried out', 'was conducted', 'seems', 
                 'suggests', 'risk', 'implicated', 'the aim', 'to investigate',
                 'to assess', 'to study'],
        'ctd_unspecified': ctd_unspecified,
        'ctd_therapy': ctd_therapy,
        'ctd_marker': ctd_marker,
    }
    return user_lists

basic = [
    # LF_c_cause_d
    Explanation(
        name='LF_c_cause_d',
        condition="""Label true because any causal phrase is between the 
            chemical and the disease and the word 'not' is not between the 
            chemical and the disease""",
        paraphrase="""Label true because between the chemical and the disease, 
            there is a causal word and the word 'not' is not between them.""",
        candidate=6606713828167518488,
        label=1,
        semantics=('.root', ('.label', ('.bool', True), ('.and', ('.any', ('.map', ('.in', ('.extract_text', ('.between', ('.list', ('.arg', ('.int', 1)), ('.arg', ('.int', 2)))))), ('.user_list', ('.string', u'causal')))), ('.not', ('.call', ('.in', ('.extract_text', ('.between', ('.list', ('.arg', ('.int', 1)), ('.arg', ('.int', 2)))))), ('.string', u'not'))))))),
    # LF_c_d
    Explanation(
        name='LF_c_d',
        condition="Label true because the disease is immediately after the chemical",
        paraphrase="""Label true because the disease is immediately preceded by the chemical.""",
        candidate=4911918761913559389,
        label=1,
        semantics=('.root', ('.label', ('.bool', True), ('.call', ('.in', ('.extract_text', ('.right', ('.arg', ('.int', 1)), ('.string', '.eq'), ('.int', 1), ('.string', 'words')))), ('.arg_to_string', ('.arg', ('.int', 2))))))),
    # LF_c_induced_d
    Explanation(
        name='LF_c_induced_d',
        condition="""Label true because the disease is immediately after the 
            chemical and 'induc' or 'assoc' is in the chemical""",
        paraphrase="""Label true because the disease is immediately preceded by the chemical, 
            and the chemical name contains an "induc" or "assoc" root.""",
        candidate=6618773943628884463,
        label=1,
        semantics=('.root', ('.label', ('.bool', True), ('.and', ('.call', ('.in', ('.extract_text', ('.right', ('.arg', ('.int', 1)), ('.string', '.eq'), ('.int', 1), ('.string', 'words')))), ('.arg_to_string', ('.arg', ('.int', 2)))), ('.any', ('.map', ('.in', ('.arg_to_string', ('.arg', ('.int', 1)))), ('.list', ('.string', u'induc'), ('.string', u'assoc')))))))),
    # LF_c_treat_d
    Explanation(
        name='LF_c_treat_d',
        condition="""Label false because any word between the chemical and 
            the disease contains a treat word and the chemical is within 100 
            characters to the left of the disease""",
        paraphrase="""Label false because the chemical precedes the disease by no more than 100 characters, 
            and a word between the disease and the chemical contains a root in the treat dictionary.""",
        candidate=5000202430163451980,
        label=-1,
        semantics=('.root', ('.label', ('.bool', False), ('.and', ('.any', ('.map', ('.composite_or', ('.contains',), ('.user_list', ('.string', u'treat'))), ('.extract_text', ('.between', ('.list', ('.arg', ('.int', 1)), ('.arg', ('.int', 2))))))), ('.call', ('.in', ('.extract_text', ('.left', ('.arg', ('.int', 2)), ('.string', '.leq'), ('.int', 100), ('.string', 'chars')))), ('.arg_to_string', ('.arg', ('.int', 1)))))))),
    # LF_c_treat_d_wide
    Explanation(
        name='LF_c_treat_d_wide',
        condition="""Label false because any word between the chemical and 
            the disease contains a treat word and the chemical is left of the 
            disease""",
        paraphrase="""Label false because the chemical comes before the disease, 
            and a word between them contains a treat root.""",
        candidate=-5412508044020208858,
        label=-1,
        semantics=('.root', ('.label', ('.bool', False), ('.and', ('.any', ('.map', ('.composite_or', ('.contains',), ('.user_list', ('.string', u'treat'))), ('.extract_text', ('.between', ('.list', ('.arg', ('.int', 1)), ('.arg', ('.int', 2))))))), ('.call', ('.in', ('.extract_text', ('.left', ('.arg', ('.int', 2))))), ('.arg_to_string', ('.arg', ('.int', 1)))))))),
    # LF_ctd_marker_c_d
    Explanation(
        name='LF_ctd_marker_c_d',
        condition="""Label true because the disease is immediately after the 
            chemical and the pair of canonical ids of the chemical and disease 
            is in ctd_marker""",
        paraphrase="""Label true because the disease is immediately preceded by the chemical, 
            and the pair of the chemical and the disease canonical IDs appears in the ctd_marker dictionary.""",
        candidate=3829603392041554457,
        label=1,
        semantics=('.root', ('.label', ('.bool', True), ('.and', ('.call', ('.in', ('.extract_text', ('.right', ('.arg', ('.int', 1)), ('.string', '.eq'), ('.int', 1), ('.string', 'words')))), ('.arg_to_string', ('.arg', ('.int', 2)))), ('.call', ('.in', ('.user_list', ('.string', u'ctd_marker'))), ('.tuple', ('.map', ('.cid',), ('.list', ('.arg', ('.int', 1)), ('.arg', ('.int', 2)))))))))),
    # LF_ctd_therapy_treat
    Explanation(
        name='LF_ctd_therapy_treat',
        condition="""Label false because 
            (any word between the chemical and the disease contains a treat word and the chemical is left of the 
            disease)
            and 
            (the pair of canonical ids of the chemical and disease is in ctd_therapy)""",
        paraphrase="""Label false because
            (the chemical comes before the disease, and a word between them contains a treat word)
            and
            (the pair of the chemical and the disease canonical IDs appears in the ctd_therapy dictionary.)""",
        candidate=9013931201987912271,
        label=-1,
        semantics=('.root', ('.label', ('.bool', False), ('.and', ('.and', ('.any', ('.map', ('.composite_or', ('.contains',), ('.user_list', ('.string', u'treat'))), ('.extract_text', ('.between', ('.list', ('.arg', ('.int', 1)), ('.arg', ('.int', 2))))))), ('.call', ('.in', ('.extract_text', ('.left', ('.arg', ('.int', 2))))), ('.arg_to_string', ('.arg', ('.int', 1))))), ('.call', ('.in', ('.user_list', ('.string', u'ctd_therapy'))), ('.tuple', ('.map', ('.cid',), ('.list', ('.arg', ('.int', 1)), ('.arg', ('.int', 2)))))))))),
    # LF_ctd_unspecified_treat
    Explanation(
        name='LF_ctd_unspecified_treat',
        condition="""Label false because 
            (any word between the chemical and the disease contains a treat word and the chemical is left of the 
            disease)
            and 
            (the pair of canonical ids of the chemical and disease is in ctd_unspecified)""",
        paraphrase="""Label false because
            (the chemical comes before the disease, and a word between them contains a treat word)
            and
            (the pair of the chemical and the disease canonical IDs appears in the ctd_unspecified dictionary.)""",
        candidate=-6222536315024461563,
        label=-1,
        semantics=('.root', ('.label', ('.bool', False), ('.and', ('.and', ('.any', ('.map', ('.composite_or', ('.contains',), ('.user_list', ('.string', u'treat'))), ('.extract_text', ('.between', ('.list', ('.arg', ('.int', 1)), ('.arg', ('.int', 2))))))), ('.call', ('.in', ('.extract_text', ('.left', ('.arg', ('.int', 2))))), ('.arg_to_string', ('.arg', ('.int', 1))))), ('.call', ('.in', ('.user_list', ('.string', u'ctd_unspecified'))), ('.tuple', ('.map', ('.cid',), ('.list', ('.arg', ('.int', 1)), ('.arg', ('.int', 2)))))))))),
    # LF_d_following_c
    Explanation(
        name='LF_d_following_c',
        condition="""Label true because 'following' is between the disease 
            and the chemical and any word after the chemical contains a 
            procedure word""",
        paraphrase="""Label True because after the chemical is a word that contains a procedure root, 
            and the word "following" appears between the chemical and the disease.""",
        candidate=-6971513852802444953,
        label=1,
        semantics=('.root', ('.label', ('.bool', True), ('.and', ('.call', ('.in', ('.extract_text', ('.between', ('.list', ('.arg', ('.int', 2)), ('.arg', ('.int', 1)))))), ('.string', u'following')), ('.any', ('.map', ('.composite_or', ('.contains',), ('.user_list', ('.string', u'procedure'))), ('.extract_text', ('.right', ('.arg', ('.int', 1)))))))))),
    # LF_d_induced_by_c
    Explanation(
        name='LF_d_induced_by_c',
        condition="""Label True because 'induced by', 'caused by', or 'due to' 
            is between the disease and the chemical.""",
        paraphrase="""Label True because "induced by", "caused by", or "due to" 
        appears between the chemical and the disease.""",
        candidate=-6762188659294394913,
        label=1,
        semantics=('.root', ('.label', ('.bool', True), ('.any', ('.map', ('.in', ('.extract_text', ('.between', ('.list', ('.arg', ('.int', 2)), ('.arg', ('.int', 1)))))), ('.list', ('.string', u'induced by'), ('.string', u'caused by'), ('.string', u'due to'))))))),
    # LF_d_induced_by_c_tight
    Explanation(
        name='LF_d_induced_by_c_tight',
        condition="""Label True because 'induced by', 'caused by', or 'due to' 
            is between the disease and the chemical and 'by' or 'to' is 
            immediately to the left of the chemical.""",
        paraphrase="""Label true because the chemical is immediately preceded by 
            the word "by" or "to", and the words "induced by", "caused by", or "due to" 
            appear between the chemical and the disease.""",
        candidate=-8780309308829124768,
        label=1,
        semantics=('.root', ('.label', ('.bool', True), ('.and', ('.any', ('.map', ('.in', ('.extract_text', ('.between', ('.list', ('.arg', ('.int', 2)), ('.arg', ('.int', 1)))))), ('.list', ('.string', u'induced by'), ('.string', u'caused by'), ('.string', u'due to')))), ('.any', ('.map', ('.in', ('.extract_text', ('.left', ('.arg', ('.int', 1)), ('.string', '.eq'), ('.int', 1), ('.string', 'words')))), ('.list', ('.string', u'by'), ('.string', u'to')))))))),
    # LF_d_treat_c
    Explanation(
        name='LF_d_treat_c',
        condition="""Label false because any word between the chemical and 
            the disease contains a treat word and the chemical is within 100
            characters to the right of the disease""",
        paraphrase="""Label false because the disease precedes the chemical by no more than 100 characters, 
            and at least word between them contains a treat word.""",
        candidate=192760603909025752,
        label=-1,
        semantics=('.root', ('.label', ('.bool', False), ('.and', ('.any', ('.map', ('.composite_or', ('.contains',), ('.user_list', ('.string', u'treat'))), ('.extract_text', ('.between', ('.list', ('.arg', ('.int', 1)), ('.arg', ('.int', 2))))))), ('.call', ('.in', ('.extract_text', ('.right', ('.arg', ('.int', 2)), ('.string', '.leq'), ('.int', 100), ('.string', 'chars')))), ('.arg_to_string', ('.arg', ('.int', 1)))))))),
    # LF_develop_d_following_c
    Explanation(
        name='LF_develop_d_following_c',
        condition="""Label true because any word before the chemical contains 
            'develop' and 'following' is between the disease and the chemical""",
        paraphrase="""Label true because a word containing 'develop' appears somewhere before the chemical, 
            and the word 'following' is between the disease and the chemical.""",
        candidate=-1817051214703978965,
        label=1,
        semantics=('.root', ('.label', ('.bool', True), ('.and', ('.any', ('.map', ('.contains', ('.string', u'develop')), ('.extract_text', ('.left', ('.arg', ('.int', 1)))))), ('.call', ('.in', ('.extract_text', ('.between', ('.list', ('.arg', ('.int', 2)), ('.arg', ('.int', 1)))))), ('.string', u'following')))))),
    # LF_far_c_d
    Explanation(
        name='LF_far_c_d',
        condition="""Label false if the disease is more than 100 characters
            to the right of the chemical.""",
        paraphrase="""Label false because the chemical appears more than 100 characters before the disease.""",
        candidate=6240026992471976183,
        label=-1,
        semantics=('.root', ('.label', ('.bool', False), ('.call', ('.in', ('.extract_text', ('.right', ('.arg', ('.int', 1)), ('.string', '.gt'), ('.int', 100), ('.string', 'chars')))), ('.arg_to_string', ('.arg', ('.int', 2))))))),
    # LF_far_d_c
    Explanation(
        name='LF_far_d_c',
        condition="""Label false if the chemical is more than 100 characters
            to the right of the disease.""",
        paraphrase="""Label false because the disease appears more than 100 characters before the chemical.""",
        candidate=-5736847953411058109,
        label=-1,
        semantics=('.root', ('.label', ('.bool', False), ('.call', ('.in', ('.extract_text', ('.right', ('.arg', ('.int', 2)), ('.string', '.gt'), ('.int', 100), ('.string', 'chars')))), ('.arg_to_string', ('.arg', ('.int', 1))))))),
    # LF_improve_before_disease
    Explanation(
        name='LF_improve_before_disease',
        condition="""Label false if any word before the disease starts with 'improv'""",
        paraphrase="""Label false because a word starting with "improv" appears before the chemical.""",
        candidate=4358774324608031121,
        label=-1,
        semantics=('.root', ('.label', ('.bool', False), ('.any', ('.map', ('.startswith', ('.string', u'improv')), ('.extract_text', ('.left', ('.arg', ('.int', 2))))))))),
    # LF_in_ctd_unspecified
    Explanation(
        name='LF_in_ctd_unspecified',
        condition="""Label false if the pair of canonical ids of the chemical 
            and disease is in ctd_unspecified""",
        paraphrase="""Label false because the pair of canonical IDs of the 
            chemical and the disease are in the ctd_unspecified dictionary.""",
        candidate=-5889490471583847150,
        label=-1,
        semantics=('.root', ('.label', ('.bool', False), ('.call', ('.in', ('.user_list', ('.string', u'ctd_unspecified'))), ('.tuple', ('.map', ('.cid',), ('.list', ('.arg', ('.int', 1)), ('.arg', ('.int', 2))))))))),
    # LF_in_ctd_therapy
    Explanation(
        name='LF_in_ctd_therapy',
        condition="""Label false if the pair of canonical ids of the chemical 
            and disease is in ctd_therapy""",
        paraphrase="""Label false because the pair of canonical IDs of the 
            chemical and the disease are in the ctd_therapy dictionary.""",
        candidate=1928996051652884359,
        label=-1,
        semantics=('.root', ('.label', ('.bool', False), ('.call', ('.in', ('.user_list', ('.string', u'ctd_therapy'))), ('.tuple', ('.map', ('.cid',), ('.list', ('.arg', ('.int', 1)), ('.arg', ('.int', 2))))))))),
    # LF_in_ctd_marker
    Explanation(
        name='LF_in_ctd_marker',
        condition="""Label true if the pair of canonical ids of the chemical 
            and disease is in ctd_marker""",
        paraphrase="""Label true because the pair of canonical IDs of the 
            chemical and the disease are in the ctd_marker dictionary.""",
        candidate=-5889490471583847150,
        label=1,
        semantics=('.root', ('.label', ('.bool', True), ('.call', ('.in', ('.user_list', ('.string', u'ctd_marker'))), ('.tuple', ('.map', ('.cid',), ('.list', ('.arg', ('.int', 1)), ('.arg', ('.int', 2))))))))),
    # LF_in_patient_with
    Explanation(
        name='LF_in_patient_with',
        condition="""Label false if any patient phrase is within four words 
            before the disease""",
        paraphrase="""Label false because a patient phrase comes no more than 
            four words before the disease.""",
        candidate=-1516295839967862351,
        label=-1,
        semantics=('.root', ('.label', ('.bool', False), ('.any', ('.map', ('.in', ('.extract_text', ('.left', ('.arg', ('.int', 2)), ('.string', '.leq'), ('.int', 4), ('.string', 'words')))), ('.user_list', ('.string', u'patient'))))))),
    # LF_induce
    Explanation(
        name='LF_induce',
        condition="""Label true because any word between the chemical and the 
            disease contains 'induc'""",
        paraphrase="""Label true because a word between the chemical and the 
            disease contains "induc".""",
        candidate=-6270620972052954916,
        label=1,
        semantics=('.root', ('.label', ('.bool', True), ('.any', ('.map', ('.contains', ('.string', u'induc')), ('.extract_text', ('.between', ('.list', ('.arg', ('.int', 1)), ('.arg', ('.int', 2)))))))))),
    # LF_induce_name
    Explanation(
        name='LF_induce_name',
        condition="Label True because the chemical contains 'induc'",
        paraphrase="""Label True because the chemical contains "induc".""",
        candidate=3240895064801201846,
        label=1,
        semantics=('.root', ('.label', ('.bool', True), ('.call', ('.contains', ('.string', u'induc')), ('.arg_to_string', ('.arg', ('.int', 1))))))),
    # LF_induced_other
    Explanation(
        name='LF_induced_other',
        condition="""Label false if any word between the chemical and the 
            disease ends with 'induced'""",
        paraphrase="""Label false because a word between the chemical and the 
            disease ends with "induced".""",
        candidate=2418695948208481836,
        label=-1,
        semantics=('.root', ('.label', ('.bool', False), ('.any', ('.map', ('.endswith', ('.string', u'induced')), ('.extract_text', ('.between', ('.list', ('.arg', ('.int', 1)), ('.arg', ('.int', 2)))))))))),
    # LF_level
    Explanation(
        name='LF_level',
        condition="""Label false because 'level' comes after the chemical""",
        paraphrase="""Label false because the word "level" comes after the chemical.""",
        candidate=7137204889488246129,
        label=-1,
        semantics=('.root', ('.label', ('.bool', False), ('.call', ('.in', ('.extract_text', ('.right', ('.arg', ('.int', 1))))), ('.string', u'level'))))),
    # LF_measure
    Explanation(
        name='LF_measure',
        condition="""Label false because any word before the chemical starts 
            with 'measur'""",
        paraphrase="""Label false because a word before the chemical starts 
            with "measur".""",
        candidate=4105760717408167415,
        label=-1,
        semantics=('.root', ('.label', ('.bool', False), ('.any', ('.map', ('.startswith', ('.string', u'measur')), ('.extract_text', ('.left', ('.arg', ('.int', 1))))))))),
    # LF_neg_d
    Explanation(
        name='LF_neg_d',
        condition="""Label false because 'none', 'not', or 'no' is within 30 
            characters to the left of the disease""",
        paraphrase="""Label false because "none", "not", or "no" precedes the 
            disease by no more than 30 characters.
        """,
        candidate=7708285380769583739,
        label=-1,
        semantics=('.root', ('.label', ('.bool', False), ('.any', ('.map', ('.in', ('.extract_text', ('.left', ('.arg', ('.int', 2)), ('.string', '.leq'), ('.int', 30), ('.string', 'chars')))), ('.list', ('.string', u'none'), ('.string', u'not'), ('.string', u'no'))))))),
    # LF_risk_d
    Explanation(
        name='LF_risk_d',
        condition="""Label true because the phrase 'risk of' occurs before 
            the disease""",
        paraphrase="""Label true because "risk of" comes before the disease.""",
        candidate=4499078534190694908,
        label=1,
        semantics=('.root', ('.label', ('.bool', True), ('.call', ('.in', ('.extract_text', ('.left', ('.arg', ('.int', 2))))), ('.string', u'risk of'))))),
    # LF_treat_d
    Explanation(
        name='LF_treat_d',
        condition="""Label false because at least one treat word is less than
            50 characters before the disease""",
        paraphrase="""Label false because there is at least one treat word 
            no more than 50 characters before the disease.""",
        candidate=-4670194985477947653,
        label=-1,
        semantics=('.root', ('.label', ('.bool', False), ('.call', ('.geq', ('.int', 1)), ('.sum', ('.map', ('.in', ('.extract_text', ('.left', ('.arg', ('.int', 2)), ('.string', '.lt'), ('.int', 50), ('.string', 'chars')))), ('.user_list', ('.string', u'treat')))))))),
    # LF_uncertain
    Explanation(
        name='LF_uncertain',
        condition="""Label false if any word before the chemical starts with 
            an uncertain word""",
        paraphrase="""Label false because the chemical is preceded by a word 
            that starts with a word that appears in the uncertain dictionary.
        """,
        candidate=1589307577177419147,
        label=-1,
        semantics=('.root', ('.label', ('.bool', False), ('.any', ('.map', ('.composite_or', ('.startswith',), ('.user_list', ('.string', u'uncertain'))), ('.extract_text', ('.left', ('.arg', ('.int', 1))))))))),
    # LF_weak_assertions
    Explanation(
        name='LF_weak_assertions',
        condition="""Label false because at least one weak phrase is in 
            the sentence""",
        paraphrase="""Label false because at least one weak phrase is in the sentence.""",
        candidate=8898005229761872427,
        label=-1,
        semantics=('.root', ('.label', ('.bool', False), ('.call', ('.geq', ('.int', 1)), ('.sum', ('.map', ('.in', ('.extract_text', ('.sentence',))), ('.user_list', ('.string', u'weak')))))))),
]

redundant = [
    # LF_ctd_unspecified_induce
    Explanation(
        name='LF_ctd_unspecified_induce',
        condition="""Label True because
            (
                (the disease is immediately after the chemical and the chemical contains 'induc' or 'assoc')
                or
                ('induced by', 'caused by', or 'due to' is between the disease and the chemical)
            )
            and 
            (the pair of canonical ids of the chemical and disease is in ctd_unspecified)""",
        paraphrase="""Label true because 
            (
                (the disease is immediately preceded by the chemical, and the chemical name contains an "induc" or "assoc" root)
                or 
                ("induced by", "caused by", or "due to" appears between the chemical and the disease)
            )
            and
            (the pair of the chemical and the disease canonical IDs appears in the ctd_unspecified dictionary.)""",
        candidate=-249729854237013355,
        label=1,
        semantics=('.root', ('.label', ('.bool', True), ('.and', ('.or', ('.and', ('.call', ('.in', ('.extract_text', ('.right', ('.arg', ('.int', 1)), ('.string', '.eq'), ('.int', 1), ('.string', 'words')))), ('.arg_to_string', ('.arg', ('.int', 2)))), ('.call', ('.composite_or', ('.contains',), ('.list', ('.string', u'induc'), ('.string', u'assoc'))), ('.arg_to_string', ('.arg', ('.int', 1))))), ('.any', ('.map', ('.in', ('.extract_text', ('.between', ('.list', ('.arg', ('.int', 2)), ('.arg', ('.int', 1)))))), ('.list', ('.string', u'induced by'), ('.string', u'caused by'), ('.string', u'due to'))))), ('.call', ('.in', ('.user_list', ('.string', u'ctd_unspecified'))), ('.tuple', ('.map', ('.cid',), ('.list', ('.arg', ('.int', 1)), ('.arg', ('.int', 2)))))))))),  
    # LF_ctd_marker_induce
    Explanation(
        name='LF_ctd_marker_induce',
        condition="""Label True because
            (
                (the disease is immediately after the chemical and the chemical contains 'induc' or 'assoc')
                or
                ('induced by', 'caused by', or 'due to' is between the disease and the chemical)
            )
            and 
            (the pair of canonical ids of the chemical and disease is in ctd_marker)""",
        paraphrase="""Label true because 
            (
                (the disease is immediately preceded by the chemical, and the chemical name contains an "induc" or "assoc" root)
                or 
                ("induced by", "caused by", or "due to" appears between the chemical and the disease)
            )
            and
            (the pair of the chemical and the disease canonical IDs appears in the ctd_marker dictionary.)""",
        candidate=-305419566691337972,
        label=1,
        semantics=('.root', ('.label', ('.bool', True), ('.and', ('.or', ('.and', ('.call', ('.in', ('.extract_text', ('.right', ('.arg', ('.int', 1)), ('.string', '.eq'), ('.int', 1), ('.string', 'words')))), ('.arg_to_string', ('.arg', ('.int', 2)))), ('.call', ('.composite_or', ('.contains',), ('.list', ('.string', u'induc'), ('.string', u'assoc'))), ('.arg_to_string', ('.arg', ('.int', 1))))), ('.any', ('.map', ('.in', ('.extract_text', ('.between', ('.list', ('.arg', ('.int', 2)), ('.arg', ('.int', 1)))))), ('.list', ('.string', u'induced by'), ('.string', u'caused by'), ('.string', u'due to'))))), ('.call', ('.in', ('.user_list', ('.string', u'ctd_marker'))), ('.tuple', ('.map', ('.cid',), ('.list', ('.arg', ('.int', 1)), ('.arg', ('.int', 2)))))))))),          
]

advanced = [
    # LF_closer_chem
    Explanation(
        name='LF_closer_chem',
        condition=None,
        paraphrase=None,
        candidate=-1954799400282697253,
        label=-1,
        semantics=None),
    # LF_closer_dis
    Explanation(
        name='LF_closer_dis',
        condition=None,
        paraphrase=None,
        candidate=-130640710948826159,
        label=-1,
        semantics=None),
]

explanations = basic # + redundant + advanced

# DEPRECATED
# def load_examples(domain, candidates):
#     if domain == 'test':
#         examples = test_examples + 
#     elif domain == 'spouse':
#         examples = spouse_examples
#     elif domain == 'cdr':
#         examples = cdr_examples
#     else:
#         raise Exception("Invalid example set provided.")
    
#     candidate_dict = {hash(c) : c for c in candidates}
#     for example in examples:
#         if example.candidate and not isinstance(example.candidate, tuple):
#             example.candidate = candidate_dict[hash(example.candidate)]
#     # assert(example.candidate is not None for example in examples)
#     return examples