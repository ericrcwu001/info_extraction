#!/usr/bin/python3

from os import walk
from os.path import splitext, join, exists
from absl import flags, app
from tqdm import tqdm
import json
from rag import RAG

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('input_dir', default = None, help = 'path to directory containing pdfs')
  flags.DEFINE_boolean('run_locally', default = False, help = 'whether run LLM locally')
  flags.DEFINE_string('output_json', default = 'output.json', help = 'path to output json')

def main(unused_argv):
  content = list()
  for root, dirs, files in tqdm(walk(FLAGS.input_dir)):
    for f in files:
      print('处理%s' % f)
      stem, ext = splitext(f)
      if ext.lower() not in ['.htm', '.html', '.txt']: continue
      rag = RAG(join(root, f), locally = FLAGS.run_locally)
      neg, _ = rag.query("负极材料属于['碳基','硅基','锂金属或锂合金','氧化物','硫化物']中的哪一类？如果材料不在这些类别中，给出具体材料名称。")
      pos, _ = rag.query("正极材料属于['氧化钴锂','氧化镍锂','氧化锰锂','磷酸亚铁锂','硫化物']中的哪一类？如果材料不在这些类别中，给出具体材料名称。")
      electrolyte, _ = rag.query("电解质属于['聚合物','氧化物','硫化物']中的哪一类？如果电解质不在这些类别中，给出具体电解质名称。")
      structure, _ = rag.query("电池结构属于['wound cell','stacked cell']中的哪一类？如果电池结构不在这些类别中，给出具体电池结构名称。")
      company, _ = rag.query("公司名字是什么？")
      content.append({"专利":f,"公司":company,"正极材料":pos,"负极材料":neg,"电解质":electrolyte,"电池结构":structure})
  with open(FLAGS.output_json, 'w', encoding = 'utf-8') as f:
    f.write(json.dumps(content, indent = 2, ensure_ascii = False))

if __name__ == "__main__":
  add_options()
  app.run(main)

