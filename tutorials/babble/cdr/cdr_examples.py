import bz2
import cPickle
import os

from snorkel.contrib.babble import Explanation, link_explanation_candidates

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
        # original_condition="""Label true because any causal phrase is between the 
        #     chemical and the disease and the word 'not' is not between the 
        #     chemical and the disease""",
        condition="""Label true because between the chemical and the disease, 
            there is a causal word and the word 'not' is not between them.""",
        candidate="6286738::span:1020:1029~~6286738::span:1077:1086",
        label=True,
        semantics=('.root', ('.label', ('.bool', True), ('.and', ('.any', ('.map', ('.in', ('.extract_text', ('.between', ('.list', ('.arg', ('.int', 1)), ('.arg', ('.int', 2)))))), ('.user_list', ('.string', u'causal')))), ('.not', ('.call', ('.in', ('.extract_text', ('.between', ('.list', ('.arg', ('.int', 1)), ('.arg', ('.int', 2)))))), ('.string', u'not'))))))),
    # LF_c_d
    Explanation(
        name='LF_c_d',
        # original_condition="Label true because the disease is immediately after the chemical",
        condition="""Label true because the disease is immediately preceded by the chemical.""",
        candidate="10669626::span:864:879~~10669626::span:881:900",
        label=True,
        semantics=('.root', ('.label', ('.bool', True), ('.call', ('.in', ('.extract_text', ('.right', ('.arg', ('.int', 1)), ('.string', '.eq'), ('.int', 1), ('.string', 'words')))), ('.arg_to_string', ('.arg', ('.int', 2))))))),
    # LF_c_induced_d
    Explanation(
        name='LF_c_induced_d',
        # original_condition="""Label true because the disease is immediately after the 
        #     chemical and 'induc' or 'assoc' is in the chemical""",
        condition="""Label true because the disease is immediately preceded by the chemical, 
            and the chemical name contains an "induc" or "assoc" root.""",
        candidate="6888657::span:20:35~~6888657::span:37:59",
        label=True,
        semantics=('.root', ('.label', ('.bool', True), ('.and', ('.call', ('.in', ('.extract_text', ('.right', ('.arg', ('.int', 1)), ('.string', '.eq'), ('.int', 1), ('.string', 'words')))), ('.arg_to_string', ('.arg', ('.int', 2)))), ('.any', ('.map', ('.in', ('.arg_to_string', ('.arg', ('.int', 1)))), ('.list', ('.string', u'induc'), ('.string', u'assoc')))))))),
    # LF_c_treat_d
    Explanation(
        name='LF_c_treat_d',
        # original_condition="""Label false because any word between the chemical and 
        #     the disease contains a treat word and the chemical is within 100 
        #     characters to the left of the disease""",
        condition="""Label false because the chemical precedes the disease by no more than 100 characters, 
            and a word between the disease and the chemical contains a root in the treat dictionary.""",
        candidate="2549018::span:93:102~~2549018::span:158:173",
        label=False,
        semantics=('.root', ('.label', ('.bool', False), ('.and', ('.any', ('.map', ('.composite_or', ('.contains',), ('.user_list', ('.string', u'treat'))), ('.extract_text', ('.between', ('.list', ('.arg', ('.int', 1)), ('.arg', ('.int', 2))))))), ('.call', ('.in', ('.extract_text', ('.left', ('.arg', ('.int', 2)), ('.string', '.leq'), ('.int', 100), ('.string', 'chars')))), ('.arg_to_string', ('.arg', ('.int', 1)))))))),
    # LF_c_treat_d_wide
    Explanation(
        name='LF_c_treat_d_wide',
        # original_condition="""Label false because any word between the chemical and 
        #     the disease contains a treat word and the chemical is left of the 
        #     disease""",
        condition="""Label false because the chemical comes before the disease, 
            and a word between them contains a treat root.""",
        candidate="2549018::span:93:102~~2549018::span:125:153",
        label=False,
        semantics=('.root', ('.label', ('.bool', False), ('.and', ('.any', ('.map', ('.composite_or', ('.contains',), ('.user_list', ('.string', u'treat'))), ('.extract_text', ('.between', ('.list', ('.arg', ('.int', 1)), ('.arg', ('.int', 2))))))), ('.call', ('.in', ('.extract_text', ('.left', ('.arg', ('.int', 2))))), ('.arg_to_string', ('.arg', ('.int', 1)))))))),
    # LF_ctd_marker_c_d
    Explanation(
        name='LF_ctd_marker_c_d',
        # original_condition="""Label true because the disease is immediately after the 
        #     chemical and the pair of canonical ids of the chemical and disease 
        #     is in ctd_marker""",
        condition="""Label true because the disease is immediately preceded by the chemical, 
            and the pair of the chemical and the disease canonical IDs appears in the ctd_marker dictionary.""",
        candidate="1967484::span:456:472~~1967484::span:474:489",
        label=True,
        semantics=('.root', ('.label', ('.bool', True), ('.and', ('.call', ('.in', ('.extract_text', ('.right', ('.arg', ('.int', 1)), ('.string', '.eq'), ('.int', 1), ('.string', 'words')))), ('.arg_to_string', ('.arg', ('.int', 2)))), ('.call', ('.in', ('.user_list', ('.string', u'ctd_marker'))), ('.tuple', ('.map', ('.cid',), ('.list', ('.arg', ('.int', 1)), ('.arg', ('.int', 2)))))))))),
    # LF_ctd_therapy_treat
    Explanation(
        name='LF_ctd_therapy_treat',
        # original_condition="""Label false because 
        #     (any word between the chemical and the disease contains a treat word and the chemical is left of the 
        #     disease)
        #     and 
        #     (the pair of canonical ids of the chemical and disease is in ctd_therapy)""",
        condition="""Label false because
            (the chemical comes before the disease, and a word between them contains a treat word)
            and
            (the pair of the chemical and the disease canonical IDs appears in the ctd_therapy dictionary.)""",
        candidate="7834920::span:70:77~~7834920::span:91:112",
        label=False,
        semantics=('.root', ('.label', ('.bool', False), ('.and', ('.and', ('.any', ('.map', ('.composite_or', ('.contains',), ('.user_list', ('.string', u'treat'))), ('.extract_text', ('.between', ('.list', ('.arg', ('.int', 1)), ('.arg', ('.int', 2))))))), ('.call', ('.in', ('.extract_text', ('.left', ('.arg', ('.int', 2))))), ('.arg_to_string', ('.arg', ('.int', 1))))), ('.call', ('.in', ('.user_list', ('.string', u'ctd_therapy'))), ('.tuple', ('.map', ('.cid',), ('.list', ('.arg', ('.int', 1)), ('.arg', ('.int', 2)))))))))),
    # LF_ctd_unspecified_treat
    Explanation(
        name='LF_ctd_unspecified_treat',
        # original_condition="""Label false because 
        #     (any word between the chemical and the disease contains a treat word and the chemical is left of the 
        #     disease)
        #     and 
        #     (the pair of canonical ids of the chemical and disease is in ctd_unspecified)""",
        condition="""Label false because
            (the chemical comes before the disease, and a word between them contains a treat word)
            and
            (the pair of the chemical and the disease canonical IDs appears in the ctd_unspecified dictionary.)""",
        candidate='10669626::span:126:134~~10669626::span:288:300',
        label=False,
        semantics=('.root', ('.label', ('.bool', False), ('.and', ('.and', ('.any', ('.map', ('.composite_or', ('.contains',), ('.user_list', ('.string', u'treat'))), ('.extract_text', ('.between', ('.list', ('.arg', ('.int', 1)), ('.arg', ('.int', 2))))))), ('.call', ('.in', ('.extract_text', ('.left', ('.arg', ('.int', 2))))), ('.arg_to_string', ('.arg', ('.int', 1))))), ('.call', ('.in', ('.user_list', ('.string', u'ctd_unspecified'))), ('.tuple', ('.map', ('.cid',), ('.list', ('.arg', ('.int', 1)), ('.arg', ('.int', 2)))))))))),
    # LF_d_following_c
    Explanation(
        name='LF_d_following_c',
        # original_condition="""Label true because 'following' is between the disease 
        #     and the chemical and any word after the chemical contains a 
        #     procedure word""",
        condition="""Label True because after the chemical is a word that contains a procedure root, 
            and the word "following" appears between the chemical and the disease.""",
        candidate="11256525::span:36:43~~11256525::span:19:24",
        label=True,
        semantics=('.root', ('.label', ('.bool', True), ('.and', ('.call', ('.in', ('.extract_text', ('.between', ('.list', ('.arg', ('.int', 2)), ('.arg', ('.int', 1)))))), ('.string', u'following')), ('.any', ('.map', ('.composite_or', ('.contains',), ('.user_list', ('.string', u'procedure'))), ('.extract_text', ('.right', ('.arg', ('.int', 1)))))))))),
    # LF_d_induced_by_c
    Explanation(
        name='LF_d_induced_by_c',
        # original_condition="""Label True because 'induced by', 'caused by', or 'due to' 
        #     is between the disease and the chemical.""",
        condition="""Label True because "induced by", "caused by", or "due to" 
        appears between the chemical and the disease.""",
        candidate="18023325::span:1250:1259~~18023325::span:1195:1228",
        label=True,
        semantics=('.root', ('.label', ('.bool', True), ('.any', ('.map', ('.in', ('.extract_text', ('.between', ('.list', ('.arg', ('.int', 2)), ('.arg', ('.int', 1)))))), ('.list', ('.string', u'induced by'), ('.string', u'caused by'), ('.string', u'due to'))))))),
    # LF_d_induced_by_c_tight
    Explanation(
        name='LF_d_induced_by_c_tight',
        # original_condition="""Label True because 'induced by', 'caused by', or 'due to' 
        #     is between the disease and the chemical and 'by' or 'to' is 
        #     immediately to the left of the chemical.""",
        condition="""Label true because the chemical is immediately preceded by 
            the word "by" or "to", and the words "induced by", "caused by", or "due to" 
            appear between the chemical and the disease.""",
        candidate="7930386::span:173:184~~7930386::span:143:159",
        label=True,
        semantics=('.root', ('.label', ('.bool', True), ('.and', ('.any', ('.map', ('.in', ('.extract_text', ('.between', ('.list', ('.arg', ('.int', 2)), ('.arg', ('.int', 1)))))), ('.list', ('.string', u'induced by'), ('.string', u'caused by'), ('.string', u'due to')))), ('.any', ('.map', ('.in', ('.extract_text', ('.left', ('.arg', ('.int', 1)), ('.string', '.eq'), ('.int', 1), ('.string', 'words')))), ('.list', ('.string', u'by'), ('.string', u'to')))))))),
    # LF_d_treat_c
    Explanation(
        name='LF_d_treat_c',
        # original_condition="""Label false because any word between the chemical and 
        #     the disease contains a treat word and the chemical is within 100
        #     characters to the right of the disease""",
        condition="""Label false because the disease precedes the chemical by no more than 100 characters, 
            and at least word between them contains a treat word.""",
        candidate="6888657::span:161:178~~6888657::span:85:100",
        label=False,
        semantics=('.root', ('.label', ('.bool', False), ('.and', ('.any', ('.map', ('.composite_or', ('.contains',), ('.user_list', ('.string', u'treat'))), ('.extract_text', ('.between', ('.list', ('.arg', ('.int', 1)), ('.arg', ('.int', 2))))))), ('.call', ('.in', ('.extract_text', ('.right', ('.arg', ('.int', 2)), ('.string', '.leq'), ('.int', 100), ('.string', 'chars')))), ('.arg_to_string', ('.arg', ('.int', 1)))))))),
    # LF_develop_d_following_c
    Explanation(
        name='LF_develop_d_following_c',
        # original_condition="""Label true because any word before the chemical contains 
        #     'develop' and 'following' is between the disease and the chemical""",
        condition="""Label true because a word containing 'develop' appears somewhere before the chemical, 
            and the word 'following' is between the disease and the chemical.""",
        candidate="3827439::span:203:216~~3827439::span:139:157",
        label=True,
        semantics=('.root', ('.label', ('.bool', True), ('.and', ('.any', ('.map', ('.contains', ('.string', u'develop')), ('.extract_text', ('.left', ('.arg', ('.int', 1)))))), ('.call', ('.in', ('.extract_text', ('.between', ('.list', ('.arg', ('.int', 2)), ('.arg', ('.int', 1)))))), ('.string', u'following')))))),
    # LF_far_c_d
    Explanation(
        name='LF_far_c_d',
        # original_condition="""Label false if the disease is more than 100 characters
        #     to the right of the chemical.""",
        condition="""Label false because the chemical appears more than 100 characters before the disease.""",
        candidate="2572625::span:314:328~~2572625::span:432:440",
        label=False,
        semantics=('.root', ('.label', ('.bool', False), ('.call', ('.in', ('.extract_text', ('.right', ('.arg', ('.int', 1)), ('.string', '.gt'), ('.int', 100), ('.string', 'chars')))), ('.arg_to_string', ('.arg', ('.int', 2))))))),
    # LF_far_d_c
    Explanation(
        name='LF_far_d_c',
        # original_condition="""Label false if the chemical is more than 100 characters
        #     to the right of the disease.""",
        condition="""Label false because the disease appears more than 100 characters before the chemical.""",
        candidate="18023325::span:1250:1259~~18023325::span:1120:1130",
        label=False,
        semantics=('.root', ('.label', ('.bool', False), ('.call', ('.in', ('.extract_text', ('.right', ('.arg', ('.int', 2)), ('.string', '.gt'), ('.int', 100), ('.string', 'chars')))), ('.arg_to_string', ('.arg', ('.int', 1))))))),
    # LF_improve_before_disease
    Explanation(
        name='LF_improve_before_disease',
        # original_condition="""Label false if any word before the disease starts with 'improv'""",
        condition="""Label false because a word starting with "improv" appears before the chemical.""",
        candidate="9201797::span:1846:1853~~9201797::span:1640:1648",
        label=False,
        semantics=('.root', ('.label', ('.bool', False), ('.any', ('.map', ('.startswith', ('.string', u'improv')), ('.extract_text', ('.left', ('.arg', ('.int', 2))))))))),
    # LF_in_ctd_unspecified
    Explanation(
        name='LF_in_ctd_unspecified',
        # original_condition="""Label false if the pair of canonical ids of the chemical 
        #     and disease is in ctd_unspecified""",
        condition="""Label false because the pair of canonical IDs of the 
            chemical and the disease are in the ctd_unspecified dictionary.""",
        candidate="1749407::span:188:194~~1749407::span:210:236",
        label=False,
        semantics=('.root', ('.label', ('.bool', False), ('.call', ('.in', ('.user_list', ('.string', u'ctd_unspecified'))), ('.tuple', ('.map', ('.cid',), ('.list', ('.arg', ('.int', 1)), ('.arg', ('.int', 2))))))))),
    # LF_in_ctd_therapy
    Explanation(
        name='LF_in_ctd_therapy',
        # original_condition="""Label false if the pair of canonical ids of the chemical 
        #     and disease is in ctd_therapy""",
        condition="""Label false because the pair of canonical IDs of the 
            chemical and the disease are in the ctd_therapy dictionary.""",
        candidate="2234245::span:1030:1044~~2234245::span:970:977",
        label=False,
        semantics=('.root', ('.label', ('.bool', False), ('.call', ('.in', ('.user_list', ('.string', u'ctd_therapy'))), ('.tuple', ('.map', ('.cid',), ('.list', ('.arg', ('.int', 1)), ('.arg', ('.int', 2))))))))),
    # LF_in_ctd_marker
    Explanation(
        name='LF_in_ctd_marker',
        # original_condition="""Label true if the pair of canonical ids of the chemical 
        #     and disease is in ctd_marker""",
        condition="""Label true because the pair of canonical IDs of the 
            chemical and the disease are in the ctd_marker dictionary.""",
        candidate="1749407::span:188:194~~1749407::span:210:236",
        label=True,
        semantics=('.root', ('.label', ('.bool', True), ('.call', ('.in', ('.user_list', ('.string', u'ctd_marker'))), ('.tuple', ('.map', ('.cid',), ('.list', ('.arg', ('.int', 1)), ('.arg', ('.int', 2))))))))),
    # LF_in_patient_with
    Explanation(
        name='LF_in_patient_with',
        # original_condition="""Label false if any patient phrase is within four words 
        #     before the disease""",
        condition="""Label false because a patient phrase comes no more than 
            four words before the disease.""",
        candidate="4071154::span:205:216~~4071154::span:161:169",
        label=False,
        semantics=('.root', ('.label', ('.bool', False), ('.any', ('.map', ('.in', ('.extract_text', ('.left', ('.arg', ('.int', 2)), ('.string', '.leq'), ('.int', 4), ('.string', 'words')))), ('.user_list', ('.string', u'patient'))))))),
    # LF_induce
    Explanation(
        name='LF_induce',
        # original_condition="""Label true because any word between the chemical and the 
        #     disease contains 'induc'""",
        condition="""Label true because a word between the chemical and the 
            disease contains "induc".""",
        candidate="7881871::span:158:182~~7881871::span:198:208",
        label=True,
        semantics=('.root', ('.label', ('.bool', True), ('.any', ('.map', ('.contains', ('.string', u'induc')), ('.extract_text', ('.between', ('.list', ('.arg', ('.int', 1)), ('.arg', ('.int', 2)))))))))),
    # LF_induce_name
    Explanation(
        name='LF_induce_name',
        # original_condition="Label True because the chemical contains 'induc'",
        condition="""Label True because the chemical contains "induc".""",
        candidate="1749407::span:412:426~~1749407::span:318:338",
        label=True,
        semantics=('.root', ('.label', ('.bool', True), ('.call', ('.contains', ('.string', u'induc')), ('.arg_to_string', ('.arg', ('.int', 1))))))),
    # LF_induced_other
    Explanation(
        name='LF_induced_other',
        # original_condition="""Label false if any word between the chemical and the 
        #     disease ends with 'induced'""",
        condition="""Label false because a word between the chemical and the 
            disease ends with "induced".""",
        candidate="7881871::span:93:98~~7881871::span:198:208",
        label=False,
        semantics=('.root', ('.label', ('.bool', False), ('.any', ('.map', ('.endswith', ('.string', u'induced')), ('.extract_text', ('.between', ('.list', ('.arg', ('.int', 1)), ('.arg', ('.int', 2)))))))))),
    # LF_level
    Explanation(
        name='LF_level',
        # original_condition="""Label false because 'level' comes after the chemical""",
        condition="""Label false because the word "level" comes after the chemical.""",
        candidate="16160878::span:1414:1427~~16160878::span:1349:1363",
        label=False,
        semantics=('.root', ('.label', ('.bool', False), ('.call', ('.in', ('.extract_text', ('.right', ('.arg', ('.int', 1))))), ('.string', u'level'))))),
    # LF_measure
    Explanation(
        name='LF_measure',
        # original_condition="""Label false because any word before the chemical starts 
        #     with 'measur'""",
        condition="""Label false because a word before the chemical starts 
            with "measur".""",
        candidate="20003049::span:1144:1153~~20003049::span:1274:1292",
        label=False,
        semantics=('.root', ('.label', ('.bool', False), ('.any', ('.map', ('.startswith', ('.string', u'measur')), ('.extract_text', ('.left', ('.arg', ('.int', 1))))))))),
    # LF_neg_d
    Explanation(
        name='LF_neg_d',
        # original_condition="""Label false because 'none', 'not', or 'no' is within 30 
        #     characters to the left of the disease""",
        condition="""Label false because "none", "not", or "no" precedes the 
            disease by no more than 30 characters.
        """,
        candidate="2572625::span:353:363~~2572625::span:432:440",
        label=False,
        semantics=('.root', ('.label', ('.bool', False), ('.any', ('.map', ('.in', ('.extract_text', ('.left', ('.arg', ('.int', 2)), ('.string', '.leq'), ('.int', 30), ('.string', 'chars')))), ('.list', ('.string', u'none'), ('.string', u'not'), ('.string', u'no'))))))),
    # LF_risk_d
    Explanation(
        name='LF_risk_d',
        # original_condition="""Label true because the phrase 'risk of' occurs before 
        #     the disease""",
        condition="""Label true because "risk of" comes before the disease.""",
        candidate="11752354::span:2111:2129~~11752354::span:2056:2076",
        label=True,
        semantics=('.root', ('.label', ('.bool', True), ('.call', ('.in', ('.extract_text', ('.left', ('.arg', ('.int', 2))))), ('.string', u'risk of'))))),
    # LF_treat_d
    Explanation(
        name='LF_treat_d',
        # original_condition="""Label false because at least one treat word is less than
        #     50 characters before the disease""",
        condition="""Label false because there is at least one treat word 
            no more than 50 characters before the disease.""",
        candidate="6286738::span:1107:1116~~6286738::span:1044:1053",
        label=False,
        semantics=('.root', ('.label', ('.bool', False), ('.call', ('.geq', ('.int', 1)), ('.sum', ('.map', ('.in', ('.extract_text', ('.left', ('.arg', ('.int', 2)), ('.string', '.lt'), ('.int', 50), ('.string', 'chars')))), ('.user_list', ('.string', u'treat')))))))),
    # LF_uncertain
    Explanation(
        name='LF_uncertain',
        # original_condition="""Label false if any word before the chemical starts with 
        #     an uncertain word""",
        condition="""Label false because the chemical is preceded by a word 
            that starts with a word that appears in the uncertain dictionary.
        """,
        candidate="10526274::span:666:668~~10526274::span:712:716",
        label=False,
        semantics=('.root', ('.label', ('.bool', False), ('.any', ('.map', ('.composite_or', ('.startswith',), ('.user_list', ('.string', u'uncertain'))), ('.extract_text', ('.left', ('.arg', ('.int', 1))))))))),
    # LF_weak_assertions
    Explanation(
        name='LF_weak_assertions',
        # original_condition="""Label false because at least one weak phrase is in 
        #     the sentence""",
        condition="""Label false because at least one weak phrase is in the sentence.""",
        candidate="7881871::span:158:182~~7881871::span:256:266",
        label=False,
        semantics=('.root', ('.label', ('.bool', False), ('.call', ('.geq', ('.int', 1)), ('.sum', ('.map', ('.in', ('.extract_text', ('.sentence',))), ('.user_list', ('.string', u'weak')))))))),
]

redundant = [
    # LF_ctd_unspecified_induce
    Explanation(
        name='LF_ctd_unspecified_induce',
        # original_condition="""Label True because
        #     (
        #         (the disease is immediately after the chemical and the chemical contains 'induc' or 'assoc')
        #         or
        #         ('induced by', 'caused by', or 'due to' is between the disease and the chemical)
        #     )
        #     and 
        #     (the pair of canonical ids of the chemical and disease is in ctd_unspecified)""",
        condition="""Label true because 
            (
                (the disease is immediately preceded by the chemical, and the chemical name contains an "induc" or "assoc" root)
                or 
                ("induced by", "caused by", or "due to" appears between the chemical and the disease)
            )
            and
            (the pair of the chemical and the disease canonical IDs appears in the ctd_unspecified dictionary.)""",
        candidate=-249729854237013355,
        label=True,
        semantics=('.root', ('.label', ('.bool', True), ('.and', ('.or', ('.and', ('.call', ('.in', ('.extract_text', ('.right', ('.arg', ('.int', 1)), ('.string', '.eq'), ('.int', 1), ('.string', 'words')))), ('.arg_to_string', ('.arg', ('.int', 2)))), ('.call', ('.composite_or', ('.contains',), ('.list', ('.string', u'induc'), ('.string', u'assoc'))), ('.arg_to_string', ('.arg', ('.int', 1))))), ('.any', ('.map', ('.in', ('.extract_text', ('.between', ('.list', ('.arg', ('.int', 2)), ('.arg', ('.int', 1)))))), ('.list', ('.string', u'induced by'), ('.string', u'caused by'), ('.string', u'due to'))))), ('.call', ('.in', ('.user_list', ('.string', u'ctd_unspecified'))), ('.tuple', ('.map', ('.cid',), ('.list', ('.arg', ('.int', 1)), ('.arg', ('.int', 2)))))))))),  
    # LF_ctd_marker_induce
    Explanation(
        name='LF_ctd_marker_induce',
        # original_condition="""Label True because
        #     (
        #         (the disease is immediately after the chemical and the chemical contains 'induc' or 'assoc')
        #         or
        #         ('induced by', 'caused by', or 'due to' is between the disease and the chemical)
        #     )
        #     and 
        #     (the pair of canonical ids of the chemical and disease is in ctd_marker)""",
        condition="""Label true because 
            (
                (the disease is immediately preceded by the chemical, and the chemical name contains an "induc" or "assoc" root)
                or 
                ("induced by", "caused by", or "due to" appears between the chemical and the disease)
            )
            and
            (the pair of the chemical and the disease canonical IDs appears in the ctd_marker dictionary.)""",
        candidate=-305419566691337972,
        label=True,
        semantics=('.root', ('.label', ('.bool', True), ('.and', ('.or', ('.and', ('.call', ('.in', ('.extract_text', ('.right', ('.arg', ('.int', 1)), ('.string', '.eq'), ('.int', 1), ('.string', 'words')))), ('.arg_to_string', ('.arg', ('.int', 2)))), ('.call', ('.composite_or', ('.contains',), ('.list', ('.string', u'induc'), ('.string', u'assoc'))), ('.arg_to_string', ('.arg', ('.int', 1))))), ('.any', ('.map', ('.in', ('.extract_text', ('.between', ('.list', ('.arg', ('.int', 2)), ('.arg', ('.int', 1)))))), ('.list', ('.string', u'induced by'), ('.string', u'caused by'), ('.string', u'due to'))))), ('.call', ('.in', ('.user_list', ('.string', u'ctd_marker'))), ('.tuple', ('.map', ('.cid',), ('.list', ('.arg', ('.int', 1)), ('.arg', ('.int', 2)))))))))),          
]

advanced = [
    # LF_closer_chem
    Explanation(
        name='LF_closer_chem',
        # original_condition=None,
        condition="",
        candidate=-1954799400282697253,
        label=False,
        semantics=None),
    # LF_closer_dis
    Explanation(
        name='LF_closer_dis',
        # original_condition=None,
        condition="",
        candidate=-130640710948826159,
        label=False,
        semantics=None),
]

explanations = basic # + redundant + advanced

def get_explanations(candidates):
    return link_explanation_candidates(explanations, candidates)

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