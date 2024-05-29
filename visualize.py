#!/usr/bin/python3

import matplotlib.pyplot as plt
from absl import flags, app
import json

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('input_json', default = None, help = 'path to json')

def main(unused_argv):
  with open(FLAGS.input_json, 'r') as f:
    results = json.loads(f.read())
  counts = {'聚合物':0,'氧化物':0,'硫化物':0,'其他':0}
  for res in results:
    if res['电解质'].startswith('聚合物'): counts['聚合物'] += 1
    elif res['电解质'].startswith('氧化物'): counts['氧化物'] += 1
    elif res['电解质'].startswith('硫化物'): counts['硫化物'] += 1
    else: counts['其他'] += 1
  fig, ax = plt.subplots()
  ax.pie(list(counts.values()), labels = list(counts.keys()))
  plt.savefig('output.png', format = "png")
  plt.show()

if __name__ == "__main__":
  add_options()
  app.run(main)
