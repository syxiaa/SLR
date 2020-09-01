# SNLR

* SNLR_RS algorithms is described in detail in .

* the implementation of the SNLR_RS algorithm of the Python version can be found in the "SNLR" file.

* All data used in the paper is in the compressed file "data.zip".

### Python version (python 版本):

* the implementations of the C_SNLR and FAR_SNLR algorithm are "C_SNLR.py" and "FAR_SNLR.py".

* the param "radius" is used to set neighborhood radius size in FAR_SNLR.

# Requirements (环境要求)

### Installation requirements (Python) (安装要求):

* Only need to rely on the DLL files in the "PythonVersion" file.

# Using (用法)

### python version (python 版本):
####C_SNLR algorithm:

##### Step 1: call "fit" function (调用"fit"函数)

###### Parameters (参数说明): 

* path: absolute path of th csv file of clustering data.


###### Output (输出说明): 

* data: Preprocessed data.
*Equivalence_class:Equivalence classes of conditional attributes.

##### Step 2: Execute function rough_set_attribute

###### Parameters (参数说明): 

* data:Preprocessed data.

* Equivalence_class:Equivalence classes of conditional attributes.

###### Output (输出说明): results: attribute reduction results


####FAR_SNLR algorithm:

##### Step 1: call "fit" function (调用"fit"函数)

###### Parameters (参数说明): 

* path: absolute path of th csv file of clustering data.

###### Output (输出说明): 

* data: Preprocessed data.

##### Step 2: Execute function rough_set_attribute

###### Parameters (参数说明): 

* data: data samples to be calculated

*alldata: Preprocessed data.

* radius: Neighborhood radius value

###### Output (输出说明): results: attribute reduction results

# Examples (示例):


# Doesn't work? (有疑问？)

* Please contact Hao Zhang at zhanghao_cqupt@qq.com
