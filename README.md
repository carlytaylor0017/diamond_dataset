# diamond_dataset
# Building a linear regression model to predict prices of diamonds
## Carly Wolfbrandt

### Table of Contents
1. [Exploratory Data Analysis](#eda)
    1. [Diamond Dataset](#dataset) 
    2. [Data Cleaning](#cleaning)
4. [Model Building](#model)
    1. [Linear Regression without Transformations](#notransform)
    2. [Linear Regression with Logorithmic Transformation](#transform)
5. [Future Work](#future_work)

## Exploratory Data Analysis <a name="eda"></a>

### Diamond Dataset <a name="dataset"></a>

**Table 1**: Initial diamond dataset 

|    |   price | cut     | color   | clarity   |   depth |   table |    x |    y |    z |
|---:|--------:|:--------|:--------|:----------|--------:|--------:|-----:|-----:|-----:|
|  0 |     326 | Ideal   | E       | SI2       |    61.5 |      55 | 3.95 | 3.98 | 2.43 |
|  1 |     326 | Premium | E       | SI1       |    59.8 |      61 | 3.89 | 3.84 | 2.31 |
|  2 |     327 | Good    | E       | VS1       |    56.9 |      65 | 4.05 | 4.07 | 2.31 |
|  3 |     334 | Premium | I       | VS2       |    62.4 |      58 | 4.2  | 4.23 | 2.63 |
|  4 |     335 | Good    | J       | SI2       |    63.3 |      58 | 4.34 | 4.35 | 2.75 |

### Data Cleaning <a name="cleaning"></a>

**Table 2**: Descriptive statistics of dataset summarizing the central tendency, dispersion and shape of distribution

|       |    price |      depth |       table |           x |           y |            z |
|:------|---------:|-----------:|------------:|------------:|------------:|-------------:|
| count | 40000    | 40000      | 40000       | 40000       | 40000       | 40000        |
| mean  |  3927.02 |    61.7537 |    57.4608  |     5.72918 |     5.73174 |     3.53813  |
| std   |  3982.23 |     1.43   |     2.23462 |     1.12113 |     1.12016 |     0.709047 |
| min   |   326    |    43      |    43       |     0       |     0       |     0        |
| 25%   |   949    |    61      |    56       |     4.71    |     4.72    |     2.91     |
| 50%   |  2401    |    61.8    |    57       |     5.7     |     5.71    |     3.52     |
| 75%   |  5313.25 |    62.5    |    59       |     6.54    |     6.54    |     4.0325   |
| max   | 18823    |    79      |    95       |    10.14    |    31.8     |    31.8      |

**Table 3**: Descriptive statistics summarizing the central tendency, dispersion and shape of distribution for cleaned dataset

|       |    price |       depth |       table |           x |          y |            z |
|:------|---------:|------------:|------------:|------------:|-----------:|-------------:|
| count | 39997    | 39997       | 39997       | 39997       | 39997      | 39997        |
| mean  |  3926.87 |    61.7538  |    57.46    |     5.73015 |     5.7319 |     3.53868  |
| std   |  3982.07 |     1.42997 |     2.22673 |     1.11852 |     1.1103 |     0.691385 |
| min   |   326    |    43       |    43       |     3.73    |     3.71   |     1.07     |
| 25%   |   949    |    61       |    56       |     4.71    |     4.72   |     2.91     |
| 50%   |  2401    |    61.8     |    57       |     5.7     |     5.71   |     3.53     |
| 75%   |  5313    |    62.5     |    59       |     6.54    |     6.54   |     4.03     |
| max   | 18823    |    79       |    76       |    10.14    |    10.1    |     6.43     |

**Table 4**: Replacing categorical values with ordinal values

|    |   price |   cut |   color |   clarity |   depth |   table |    x |    y |    z |
|---:|--------:|------:|--------:|----------:|--------:|--------:|-----:|-----:|-----:|
|  0 |     326 |     5 |       6 |         2 |    61.5 |      55 | 3.95 | 3.98 | 2.43 |
|  1 |     326 |     4 |       6 |         3 |    59.8 |      61 | 3.89 | 3.84 | 2.31 |
|  2 |     327 |     2 |       6 |         5 |    56.9 |      65 | 4.05 | 4.07 | 2.31 |
|  3 |     334 |     4 |       2 |         4 |    62.4 |      58 | 4.2  | 4.23 | 2.63 |
|  4 |     335 |     2 |       1 |         2 |    63.3 |      58 | 4.34 | 4.35 | 2.75 |


## Model Building <a name="model"></a>

### Linear Regression without Transformations  <a name="notransform"></a>

### Linear Regression with Logorithmic Transformation <a name="transform"></a>

## Future Work <a name="future_work"></a>








