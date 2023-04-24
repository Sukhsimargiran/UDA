

from __future__ import absolute_import
from __future__ import division
from __future__ import google_type_annotations
from __future__ import print_function
import os
from absl import app
from absl import flags

import tensorflow as tf

from utils import raw_data_utils

from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_bool(
    "separate_doc_by_newline", False, "")

flags.DEFINE_string(
    "output_data_dir",
    None, "")

flags.DEFINE_string(
    "sub_set",
    "unsup_in", "")

flags.DEFINE_string(
    "task_name",
    "IMDB", "")

flags.DEFINE_string(
    "raw_data_dir",
    "IMDB", "")

def dump_raw_examples(examples, separate_doc_by_newline):
  """dump raw examples."""
  tf.logging.info("dumpping raw examples")
  text_path = os.path.join(FLAGS.output_data_dir, "text.txt")
  label_path = os.path.join(FLAGS.output_data_dir, "label.txt")
  with tf.gfile.Open(text_path, "w") as text_ouf:
    with tf.gfile.Open(label_path, "w") as label_ouf:
      for example in examples:
        text_a = example.text_a
        text_b = example.text_b
        label = example.label
        text_ouf.write(text_a + "\n")
        if text_b is not None:
          text_ouf.write(text_b + "\n")
        if separate_doc_by_newline:
          text_ouf.write("\n")
        label_ouf.write(label + "\n")
  tf.logging.info("finished dumpping raw examples")


def main(argv):
  processor = raw_data_utils.get_processor(FLAGS.task_name)
  tf.logging.info("loading examples")
  FLAGS.output_data_dir = os.path.join(
      FLAGS.output_data_dir, FLAGS.sub_set)
  if not tf.gfile.Exists(FLAGS.output_data_dir):
    tf.gfile.MakeDirs(FLAGS.output_data_dir)
  if FLAGS.sub_set == "train":
    examples = processor.get_train_examples(FLAGS.raw_data_dir)
  elif FLAGS.sub_set.startswith("unsup"):
    examples = processor.get_unsup_examples(FLAGS.raw_data_dir, FLAGS.sub_set)
  else:
    assert False
  tf.logging.info("finished loading examples")
  tf.logging.info("examples num: {:d}".format(len(examples)))
  dump_raw_examples(examples, FLAGS.separate_doc_by_newline)

if __name__ == '__main__':
  app.run(main)
