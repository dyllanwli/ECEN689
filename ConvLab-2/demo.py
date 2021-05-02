from convlab2.nlu.jointBERT.multiwoz import BERTNLU
from convlab2.dst.rule.multiwoz import RuleDST
# policy
from convlab2.policy.rule.multiwoz import RulePolicy
from convlab2.policy.ppo.multiwoz import PPOPolicy
# nlg
from convlab2.nlg.template.multiwoz import TemplateNLG
from convlab2.nlg.sclstm.multiwoz import SCLSTM
from convlab2.dialog_agent import PipelineAgent, BiSession
from convlab2.evaluator.multiwoz_eval import MultiWozEvaluator
# This is the dataset
from pprint import pprint
import random
import numpy as np
import torch

# BERT nlu
sys_nlu = BERTNLU()
# simple rule DST
sys_dst = RuleDST()
# rule policy
sys_policy = RulePolicy()
# template NLG
sys_nlg = TemplateNLG(is_user=False)
# assemble
sys_agent = PipelineAgent(sys_nlu, sys_dst, sys_policy, sys_nlg, name='sys')
print("sys agent generated\n")

# MILU
user_nlu = BERTNLU()
# not use dst
user_dst = None
# rule policy
user_policy = RulePolicy(character='usr')
# template NLG
user_nlg = TemplateNLG(is_user=True)
# assemble
user_agent = PipelineAgent(user_nlu, user_dst, user_policy, user_nlg, name='user')
print("user agent generated\n")

evaluator = MultiWozEvaluator()
sess = BiSession(sys_agent=sys_agent, user_agent=user_agent, kb_query=None, evaluator=evaluator)
"""
MultiWozEvaluator to evaluate the performance. 
It uses the parsed dialog act input and policy output dialog act to calculate inform f1, book rate, and whether the task is success.
"""

def set_seed(r_seed):
    random.seed(r_seed)
    np.random.seed(r_seed)
    torch.manual_seed(r_seed)

set_seed(20200131)

sys_response = ''
sess.init_session()
print('init goal:')
pprint(sess.evaluator.goal)
print('-'*50)
for i in range(20):
    sys_response, user_response, session_over, reward = sess.next_turn(sys_response)
    print('user:', user_response)
    print('sys:', sys_response)
    print()
    if session_over is True:
        break
print('task success:', sess.evaluator.task_success())
print('book rate:', sess.evaluator.book_rate())
print('inform precision/recall/f1:', sess.evaluator.inform_F1())
print('-'*50)
print('final goal:')
pprint(sess.evaluator.goal)
print('='*100)

from convlab2.util.analysis_tool.analyzer import Analyzer

# if sys_nlu!=None, set use_nlu=True to collect more information
analyzer = Analyzer(user_agent=user_agent, dataset='multiwoz')

set_seed(20200131)
analyzer.comprehensive_analyze(sys_agent=sys_agent, model_name='sys_agent', total_dialog=100)
# analyzer.compare_models(agent_list=[sys_agent1, sys_agent2], model_name=['sys_agent1', 'sys_agent2'], total_dialog=100)