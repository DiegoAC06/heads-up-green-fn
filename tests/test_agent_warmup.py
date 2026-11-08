import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# fresh process: import agent.py (as the server does), then time the first flop decision
CODE = """
import time
import agent
from game import Action, AgentConfig, Game, make_deal
memory = {}
agent.new_half(AgentConfig(), memory)
game = Game(make_deal('warm', 3), button_seat=0)
game.apply(Action.CHECK_CALL)
game.apply(Action.CHECK_CALL)
obs = game.observation(1)
t = time.perf_counter()
agent.choose_action(obs, memory)
print(time.perf_counter() - t)
"""


def testFirstDecisionAfterAgentImportIsCheap():
  out = subprocess.run([sys.executable, "-c", CODE], cwd=ROOT, capture_output=True, text=True)
  assert out.returncode == 0, out.stderr[-800:]
  # cold, the first decision of a half costs about 11 ms (numpy first-use); warmed, about 1 ms
  assert float(out.stdout.strip().splitlines()[-1]) < 0.005
