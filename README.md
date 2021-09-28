# gpgpu_labs

### 1. Matrix multiplication on GPU
### 2. Matrix multiplication on GPU with shared memory
There are test results on various matrix size:

| N     | CPU time | GPU time | GPU shared time | Maximum deviation CPU and GPU | Maximum deviation GPU and shared GPU |
| :---: | :------: | :------: | :-------------: | :---------------------------: |:------------------------------------:|
| 500   | 0.179071 | 0.100058 | 0.042222        | 1.001327e-11                  |0|
| 1000  | 2.21963  | 0.677034 | 0.183351        | 1.884900e-11                  |0|
| 1500  | 8.34374  | 2.853644 | 0.739985        | 5.895543e-11                  |0|