#!/usr/bin/python3

from os import walk
from os.path import splitext, join, exists
from absl import flags, app
from tqdm import tqdm
from rag import RAG

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('input_dir', default = None, help = 'path to directory containing pdfs')
  flags.DEFINE_boolean('run_locally', default = False, help = 'whether run LLM locally')

def main(unused_argv):
  for root, dirs, files in tqdm(walk(FLAGS.input_dir)):
    for f in files:
      stem, ext = splitext(f)
      if ext.lower() not in ['.htm', '.html']: continue
      rag = RAG(join(root, f), locally = FLAGS.run_locally)
      result, _ = rag.query("负极材料属于['碳基','硅基','锂金属或锂合金','氧化物','硫化物']中的哪一类？")
      result, _ = rag.query("正极材料属于['氧化钴锂','氧化镍锂','氧化锰锂','磷酸亚铁锂','硫化物']中的哪一类？")
      result, _ = rag.query("电解质属于['聚合物','氧化物','硫化物']中的哪一类？")
      result, _ = rag.query("电池结构属于['wound cell','stacked cell']中的哪一类？")
      # TODO

if __name__ == "__main__":
  add_options()
  app.run(main)

