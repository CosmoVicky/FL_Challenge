import os
import json
import tensorflow as tf
import cifar

per_worker_batch_size = 40
tf_config = json.loads(os.environ['TF_CONFIG'])
num_workers = len(tf_config['cluster']['worker'])

strategy = tf.distribute.MultiWorkerMirroredStrategy()

global_batch_size = per_worker_batch_size * num_workers
multi_worker_dataset = cifar.dataset(global_batch_size)

model_path = '/tmp/keras-model'

def _is_chief(task_type, task_id):
  return (task_type == 'worker' and task_id == 0) or task_type is None

def _get_temp_dir(dirpath, task_id):
  base_dirpath = 'workertemp_' + str(task_id)
  temp_dir = os.path.join(dirpath, base_dirpath)
  tf.io.gfile.makedirs(temp_dir)
  return temp_dir

def write_filepath(filepath, task_type, task_id):
  dirpath = os.path.dirname(filepath)
  base = os.path.basename(filepath)
  if not _is_chief(task_type, task_id):
    dirpath = _get_temp_dir(dirpath, task_id)
  return os.path.join(dirpath, base)

task_type, task_id = (strategy.cluster_resolver.task_type,
                      strategy.cluster_resolver.task_id)
write_model_path = write_filepath(model_path, task_type, task_id)

with strategy.scope():
  multi_worker_model = cifar.build_and_compile_model()

multi_worker_model.fit(multi_worker_dataset[0], epochs=3, steps_per_epoch=30, validation_data=multi_worker_dataset[1],
    validation_freq=1)

multi_worker_model.save(write_model_path)

if not _is_chief(task_type, task_id):
  tf.io.gfile.rmtree(os.path.dirname(write_model_path))
