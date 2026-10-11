import fsspec,pyarrow.parquet as pq,json,pathlib,hashlib,datetime
root=pathlib.Path(__file__).parent
protocol=json.loads((root/'protocol.json').read_text())
meta=json.loads((root/'parquet-metadata.json').read_text())
assert sum(meta['row_groups'][0]['compressed_column_bytes'].values()) <= protocol['budget']['max_remote_column_compressed_bytes']
with fsspec.open(meta['url'],mode='rb',block_size=4194304,cache_type='blockcache') as f:
 table=pq.ParquetFile(f).read_row_group(0)
 for i in protocol['record_indices']:
  row=table.slice(i,1).to_pylist()[0]
  data=(json.dumps(row,ensure_ascii=False,sort_keys=True,indent=2)+'\n').encode()
  (root/f'record-{i}.json').write_bytes(data)
  messages=json.loads(row['messages'])
  if isinstance(messages,str):messages=json.loads(messages)
  keys=sorted(set(k for m in messages for k in m))
  print(json.dumps({'index':i,'instance_id':row['instance_id'],'model':row['model'],'traj_id':row['traj_id'],'resolved':row['resolved'],'record_bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'message_count':len(messages),'message_keys':keys,'roles':{role:sum(m.get('role')==role for m in messages) for role in sorted(set(m.get('role','') for m in messages))},'patch_head':row['patch'][:160]},ensure_ascii=False))
(root/'fetch-complete.json').write_text(json.dumps({'completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'protocol_sha256':hashlib.sha256((root/'protocol.json').read_bytes()).hexdigest(),'record_indices':protocol['record_indices']},indent=2)+'\n')
