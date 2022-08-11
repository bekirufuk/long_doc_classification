from tqdm import tqdm
p_bar = tqdm(range(9))
for i in range(9):
    print(i)
    p_bar.update(1)
