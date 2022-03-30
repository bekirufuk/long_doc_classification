import data_merger as dm
import data_processer as dp

def merge():
    merger = dm.Merger()
    print("\n----------\n DATA MERGE STARTED \n----------\n")
    ipcr = merger.load_ipcr()
    patent_file_list = merger.get_patents_list()
    for patent_file_counter, patent_dir in enumerate(patent_file_list):
        patent_year = merger.get_patent_year(patent_dir)
        chunks = merger.get_chunks(patent_dir)
        chunk_counter = 0
        for chunk in chunks:
            chunk_counter += 1
            chunk = merger.merge_chunk(chunk, ipcr, patent_file_counter+1, chunk_counter)
            merger.write_chunk(chunk, patent_year, chunk_counter)
    merger.finish()
    print("\n----------\n DATA MERGE FINISHED \n----------\n")


processer = dp.Processer()
processer.clean_not_label()
