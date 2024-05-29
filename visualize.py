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
  counts = {'polymer':0,'oxide':0,'sulfide':0,'others':0}
  for res in results:
    if res['电解质'].startswith('聚合物'): counts['polymer'] += 1
    elif res['电解质'].startswith('氧化物'): counts['oxide'] += 1
    elif res['电解质'].startswith('硫化物'): counts['sulfide'] += 1
    else: counts['others'] += 1
  fig, ax = plt.subplots()
  ax.pie(list(counts.values()), autopct='%1.1f%%')
  plt.legend(loc = 'best', labels = list(counts.keys()))
  plt.savefig('output.png', format = "png")
  plt.show()

if __name__ == "__main__":
  add_options()
  app.run(main)
