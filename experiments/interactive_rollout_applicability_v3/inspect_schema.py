import fsspec,pyarrow.parquet as pq,json,pathlib,datetime
root=pathlib.Path(__file__).parent
info=json.loads((root/'dataset-info.json').read_text());revision=info['sha']
url=f'https://huggingface.co/datasets/SWE-bench/SWE-smith-trajectories/resolve/{revision}/data/tool-00000-of-00008.parquet'
with fsspec.open(url,mode='rb',block_size=65536,cache_type='blockcache') as f:
 pf=pq.ParquetFile(f)
 r={'revision':revision,'url':url,'schema':str(pf.schema_arrow),'row_groups':[{'index':i,'rows':pf.metadata.row_group(i).num_rows,'compressed_column_bytes':{pf.metadata.row_group(i).column(j).path_in_schema:pf.metadata.row_group(i).column(j).total_compressed_size for j in range(pf.metadata.num_columns)}} for i in range(pf.num_row_groups)]}
 (root/'parquet-metadata.json').write_text(json.dumps(r,indent=2)+'\n'); print(json.dumps(r,indent=2))
