"""Synthetic metric contracts only; no benchmark answer files or network access."""
from copy import deepcopy
import json
from math import log2
import unittest

from code_rsi.v3.datasets import adapt_musique, adapt_browsecomp, adapt_bright, evaluate_answer
from code_rsi.v3.task_metrics import (TaskMetricError, score_task, support_set_metrics,
                                      aggregate_musique_full, score_bright)


def musique(group='fixture-1', answerable=True):
    row={'id':group,'question':'What name completes synthetic group '+group+'?',
         'answer':'Ada Lovelace','answer_aliases':['Augusta Ada King'],'answerable':answerable,
         'paragraphs':[{'idx':i,'title':'Synthetic '+str(i),
                        'paragraph_text':('complete ' if answerable else 'contrast ')+str(i),
                        'is_supporting':i in (1,7)} for i in range(20)]}
    return adapt_musique(row)


def prediction(task, *, answer='Ada Lovelace', supports=(1,7), answerable=True):
    return {'question_id':task['question_id'],'predicted_answer':answer,
            'predicted_support_idxs':list(supports),'predicted_answerable':answerable}


class AnswerAndSupportTests(unittest.TestCase):
    def test_alias_and_normalization_match_existing_answer_metrics(self):
        task,ref=musique()
        for text in ('The Ada, Lovelace!', 'Augusta Ada King'):
            result=score_task(prediction(task,answer=text),ref,task=task)
            self.assertEqual(result['answer_em'],1.)
            self.assertEqual(result['answer_f1'],1.)
            self.assertEqual(result['answer_f1'],evaluate_answer(text,ref,'f1'))

    def test_answer_f1_counts_repeated_tokens_and_alias_maximum(self):
        task,ref=musique(); ref['answers']=['alpha beta beta','unrelated']
        result=score_task(prediction(task,answer='alpha alpha beta'),ref,task=task)
        self.assertEqual(result['answer_em'],0.)
        self.assertAlmostEqual(result['answer_f1'],2/3)

    def test_missing_answer_cannot_match_empty_reference(self):
        task,ref=musique(); ref['answers']=['']
        self.assertEqual(score_task({'answer':''},ref)['answer_f1'],1.)
        missing=score_task({},ref)
        self.assertEqual(missing['answer_f1'],0.)
        self.assertEqual(missing['metric_status']['answer_f1'],'missing_prediction')
        failed=score_task({'answer':'','answer_usable':False},ref)
        self.assertEqual(failed['answer_em'],0.)
        self.assertEqual(failed['metric_status']['answer_em'],'host_delivery_failure')

    def test_missing_answer_reference_stays_unavailable(self):
        _,ref=musique(); ref.pop('answers')
        result=score_task('anything',ref)
        self.assertIsNone(result['answer_f1'])
        self.assertEqual(result['metric_status']['answer_f1'],'unavailable_reference')

    def test_support_indices_duplicates_and_false_positive(self):
        task,ref=musique()
        result=score_task(prediction(task,supports=(1,1)),ref,task=task)
        self.assertEqual(result['support_em'],0.)
        self.assertAlmostEqual(result['support_f1'],2/3)
        result=score_task(prediction(task,supports=(1,2)),ref,task=task)
        self.assertEqual(result['support_f1'],.5)
        result=score_task(prediction(task,supports=(1,7,999)),ref,task=task)
        self.assertAlmostEqual(result['support_f1'],.8)

    def test_citations_map_by_docid_not_count_or_position(self):
        task,ref=musique()
        ids=[task['documents'][i]['docid'] for i in (7,1)]
        pred={'answer':'Ada Lovelace','citations':[{'docid':d,'citation_id':'arbitrary'} for d in ids]}
        result=score_task(pred,ref,task=task)
        self.assertEqual(result['support_em'],1.)
        pred['citations']=[{'docid':task['documents'][0]['docid']}]
        self.assertEqual(score_task(pred,ref,task=task)['support_f1'],0.)

    def test_absent_or_empty_evidence_never_succeeds_against_nonempty_gold(self):
        task,ref=musique()
        for pred in ({'answer':'Ada Lovelace'}, {'answer':'Ada Lovelace','citations':[]}):
            result=score_task(pred,ref,task=task)
            self.assertEqual(result['answer_f1'],1.)
            self.assertEqual(result['support_f1'],0.)
            self.assertEqual(result['support_em'],0.)

    def test_unmappable_citation_is_unavailable_not_dropped(self):
        task,ref=musique()
        for citation in ({'citation_id':'e1'}, {'docid':'another-question/p/1'}):
            pred={'answer':'Ada Lovelace','citations':[citation,{'docid':ref['supporting_docids'][0]}]}
            result=score_task(pred,ref,task=task)
            self.assertIsNone(result['support_f1'])
            self.assertEqual(result['metric_status']['support_f1'],'unavailable_citation_mapping')
        result=score_task({'answer':'Ada Lovelace','citations':[{'docid':ref['supporting_docids'][0]}]},ref)
        self.assertIsNone(result['support_f1'])
        self.assertEqual(result['metric_status']['support_f1'],'unavailable_task_mapping')

    def test_unknown_annotation_is_not_a_true_empty_support_set(self):
        task,ref=musique(); ref['supporting_docids']=[]
        result=score_task({'answer':'Ada Lovelace','citations':[]},ref,task=task)
        self.assertIsNone(result['support_f1'])
        self.assertEqual(result['metric_status']['support_f1'],'unavailable_empty_annotation_ambiguous')
        ref['support_annotation_available']=True
        self.assertEqual(score_task({'answer':'Ada Lovelace','citations':[]},ref,task=task)['support_f1'],1.)
        self.assertEqual(score_task({'answer':'Ada Lovelace'},ref,task=task)['support_f1'],0.)
        ref.pop('supporting_docids')
        self.assertIsNone(score_task({'citations':[]},ref,task=task)['support_f1'])

    def test_support_empty_set_rule_is_official_when_explicit(self):
        self.assertEqual(support_set_metrics([],[]),{'support_em':1.,'support_f1':1.})
        self.assertEqual(support_set_metrics(['x'],[]),{'support_em':0.,'support_f1':0.})
        self.assertEqual(support_set_metrics(['x','x'],['x']),{'support_em':1.,'support_f1':1.})

    def test_answerability_is_explicit_and_separate_from_answer(self):
        task,ref=musique()
        pred=prediction(task,answerable=False)
        result=score_task(pred,ref,task=task)
        self.assertEqual(result['answer_f1'],1.)
        self.assertEqual(result['answerability'],0.)
        absent=score_task({'answer':'','abstained':True},ref,task=task)
        self.assertIsNone(absent['answerability'])
        with self.assertRaises(TaskMetricError):
            score_task({**pred,'predicted_answerable':'false'},ref,task=task)

    def test_unanswerable_answer_is_not_scored_or_relabelled_as_empty_gold(self):
        task,ref=musique(answerable=False)
        result=score_task(prediction(task,answer='correct-looking memorized answer',answerable=False),ref,task=task)
        self.assertEqual(result['answerability'],1.)
        for key in ('answer_em','answer_f1','support_em','support_f1'):
            self.assertIsNone(result[key])
            self.assertEqual(result['metric_status'][key],'not_applicable_unanswerable_branch')

    def test_conflicting_aliases_and_identity_fail_explicitly(self):
        task,ref=musique()
        with self.assertRaises(TaskMetricError):
            score_task({'answer':'one','predicted_answer':'two'},ref)
        with self.assertRaises(TaskMetricError):
            score_task({'predicted_support_idxs':[1],'support_docids':ref['supporting_docids']},ref,task=task)
        other,_=musique('other')
        with self.assertRaises(TaskMetricError):
            score_task('answer',ref,task=other)
        with self.assertRaises(TaskMetricError):
            score_task({'question_id':'wrong'},ref,task=task)
        with self.assertRaises(TaskMetricError):
            score_task(prediction(task),{**ref,'dataset':'unknown'})

    def test_reference_inputs_are_not_mutated_or_echoed(self):
        task,ref=musique(); ref['question_decomposition']=[{'answer':'PRIVATE_INTERMEDIATE_SENTINEL'}]
        pred=prediction(task); before=deepcopy((task,ref,pred))
        result=score_task(pred,ref,task=task)
        self.assertEqual((task,ref,pred),before)
        raw=json.dumps(result)
        self.assertNotIn('PRIVATE_INTERMEDIATE_SENTINEL',raw)
        self.assertNotIn('Ada Lovelace',raw)
        self.assertNotIn('supporting_docids',result)


class FullAggregationTests(unittest.TestCase):
    def pair(self,group='g1',wrong_unanswerable=False):
        rows=[]
        for answerable in (True,False):
            task,ref=musique(group,answerable)
            pred=prediction(task,answerable=answerable or wrong_unanswerable)
            rows.append(score_task(pred,ref,task=task))
        return rows

    def test_paired_gate_keeps_answerable_branch_only_and_macro_averages(self):
        pair1=self.pair(); pair2=self.pair('g2',wrong_unanswerable=True)
        task,ref=musique('g1'); ref['answers']=['alpha beta beta']
        pair1[0]=score_task(prediction(task,answer='alpha alpha beta'),ref,task=task)
        task,ref=musique('g2')
        pair2[0]=score_task(prediction(task,supports=(1,)),ref,task=task)
        result=aggregate_musique_full(pair2[::-1]+pair1[::-1])
        self.assertEqual(result['answer_f1'],.833)
        self.assertEqual(result['support_f1'],.833)
        self.assertEqual(result['group_answer_sufficiency_f1'],.333)
        self.assertEqual(result['group_support_sufficiency_f1'],.5)
        self.assertEqual(result['answerability'],.75)
        self.assertEqual(result['group_sufficiency'],.5)
        self.assertEqual(result['n_groups'],2)
        self.assertEqual(result['independent_unit'],'question_pair')

    def test_one_wrong_sufficiency_zeroes_group_but_not_base_answer_metric(self):
        result=aggregate_musique_full(self.pair(wrong_unanswerable=True))
        self.assertEqual(result['answer_f1'],1.)
        self.assertEqual(result['group_answer_sufficiency_f1'],0.)
        self.assertEqual(result['group_support_sufficiency_f1'],0.)

    def test_missing_metric_does_not_drop_pair_or_impute_success(self):
        rows=self.pair()+self.pair('g2')
        rows[0]['support_f1']=None; rows[0]['support_em']=None
        result=aggregate_musique_full(rows)
        self.assertIsNone(result['support_f1'])
        self.assertIsNone(result['group_support_sufficiency_f1'])
        self.assertEqual(result['n_groups'],2)
        self.assertEqual(result['answer_f1'],1.)
        rows[1]['predicted_answerable']=None; rows[1]['answerability']=None
        result=aggregate_musique_full(rows)
        self.assertIsNone(result['group_answer_sufficiency_f1'])
        self.assertIsNone(result['answerability'])

    def test_incomplete_duplicate_multi_variant_and_wrong_label_groups_rejected(self):
        rows=self.pair()
        with self.assertRaises(TaskMetricError): aggregate_musique_full(rows[:1])
        with self.assertRaises(TaskMetricError): aggregate_musique_full(rows+[rows[1]])
        third=deepcopy(rows[1]); third['question_id']='another-public-context'
        with self.assertRaises(TaskMetricError): aggregate_musique_full(rows+[third])
        broken=deepcopy(rows); broken[1]['gold_answerable']=True; broken[1]['answerability']=0.
        with self.assertRaises(TaskMetricError): aggregate_musique_full(broken)
        broken=deepcopy(rows); broken[0]['pair_group_id']=None
        with self.assertRaises(TaskMetricError): aggregate_musique_full(broken)

    def test_derived_answerability_cannot_disagree_with_labels(self):
        rows=self.pair(); rows[1]['answerability']=0.
        with self.assertRaises(TaskMetricError): aggregate_musique_full(rows)


class OtherDatasetBoundaryTests(unittest.TestCase):
    def test_browsecomp_requires_explicit_proxy_opt_in(self):
        task,ref=adapt_browsecomp({'query_id':'b','query':'synthetic BCP query','answer':'blue'},'fixed-corpus')
        result=score_task('blue',ref,task=task)
        self.assertIsNone(result['answer_f1'])
        self.assertEqual(result['metric_status']['answer_f1'],'unavailable_official_judge')
        proxy=score_task('blue',ref,task=task,allow_proxy_metrics=True)
        self.assertEqual(proxy['answer_f1'],1.)
        self.assertTrue(proxy['proxy_metrics'])
        self.assertFalse(proxy['official_judge'])
        self.assertEqual(proxy['metric_status']['answer_f1'],'proxy_rule_only')

    def bright(self):
        return adapt_bright({'id':'br','query':'synthetic retrieval query','excluded_ids':['excluded'],
                             'gold_ids':['a','b'],'gold_ids_long':['long-a']},'fixed-corpus')

    def test_bright_binary_ndcg_uses_rank_discount_and_cutoff(self):
        _,ref=self.bright()
        result=score_bright(['x','a','y','b'],ref,excluded_docids=['excluded'])
        expected=(1/log2(3)+1/log2(5))/(1+1/log2(3))
        self.assertAlmostEqual(result['ndcg_at_10'],expected)
        ranked=['negative-'+str(i) for i in range(10)]+['a','b']
        self.assertEqual(score_bright(ranked,ref,excluded_docids=[])['ndcg_at_10'],0.)
        self.assertEqual(score_bright(['long-a'],ref,excluded_docids=[],long_context=True)['ndcg_at_10'],1.)

    def test_bright_rejects_duplicates_and_exclusions_instead_of_repairing_rank(self):
        _,ref=self.bright()
        with self.assertRaises(TaskMetricError): score_bright(['a','a'],ref,excluded_docids=[])
        with self.assertRaises(TaskMetricError): score_bright(['excluded','a'],ref,excluded_docids=['excluded'])
        with self.assertRaises(TaskMetricError): score_bright(['a'],ref,excluded_docids=['a'])
        empty={**ref,'gold_ids':[]}
        self.assertIsNone(score_bright([],empty,excluded_docids=[])['ndcg_at_10'])

    def test_bright_does_not_become_a_qa_accuracy(self):
        task,ref=self.bright()
        result=score_task('arbitrary answer',ref,task=task)
        self.assertIsNone(result['answer_em'])
        self.assertEqual(result['metric_status']['answer_em'],'not_applicable_retrieval_task')


if __name__=='__main__':
    unittest.main()
