# ma_pbr_rnn
Code für die Experimente der MA.

Transformer in transformer/ von https://github.com/KasperGroesLudvigsen/influenza_transformer

IMV-LSTm in networks.py von https://github.com/KurochkinAlexey/IMV_LSTM

# abstract
Modeling of phototrophic microalgeal growth in bioreactors is often achieved by using mechanistic models for growth kinetics and complex models for radiation transport and liquid dynamics in the reactor. While these models offer good predictive performance and further understanding of underlying biological and physical processes, their construction and evaluation require comprehensive domain knowledge. Additionally models have to be adapted to fit the specific bioreactor, or get increasingly complex for more general cases. This leads to increasing computational requirements. 
    
In this thesis rnn-based models for time series prediction are examined for the task of growth prediction for the cultivation of Chlorella Vulgaris in an Industrial Plankton Bioreactor. These models have been shown to exhibit good prediction performance in many applications and novel additions offer improved interpretability. Therefore these models promise to offer an easy to use alternative to sophiosticated mechanistic models while still providing insight into the growth mechanics. 

## environment

python 3.10.6

# Name                    Version                   Build  Channel          
alembic                   1.8.1              pyhd8ed1ab_0    conda-forge    
astroid                   2.12.9          py310h5588dad_0    conda-forge    
attrs                     22.1.0             pyh71513ae_1    conda-forge    
autopage                  0.5.1              pyhd8ed1ab_0    conda-forge    
backports                 1.0                        py_2    conda-forge    
backports.functools_lru_cache 1.6.4              pyhd8ed1ab_0    conda-forge
blas                      2.116                       mkl    conda-forge    
blas-devel                3.9.0              16_win64_mkl    conda-forge    
brotli                    1.0.9                h8ffe710_7    conda-forge    
brotli-bin                1.0.9                h8ffe710_7    conda-forge    
brotlipy                  0.7.0           py310he2412df_1004    conda-forge 
bzip2                     1.0.8                h8ffe710_4    conda-forge    
ca-certificates           2022.9.14            h5b45459_0    conda-forge    
captum                    0.5.0              pyhd8ed1ab_0    conda-forge    
certifi                   2022.9.14          pyhd8ed1ab_0    conda-forge    
cffi                      1.15.1          py310hcbf9ad4_0    conda-forge    
charset-normalizer        2.1.1              pyhd8ed1ab_0    conda-forge    
cliff                     4.0.0              pyhd8ed1ab_0    conda-forge    
cmaes                     0.8.2              pyh44b312d_0    conda-forge    
cmd2                      2.4.2           py310h5588dad_0    conda-forge    
colorama                  0.4.5              pyhd8ed1ab_0    conda-forge    
colorlog                  6.7.0           py310h5588dad_0    conda-forge    
cryptography              37.0.1          py310h21b164f_0                   
cudatoolkit               11.3.1               h59b6b97_2                   
cycler                    0.11.0             pyhd8ed1ab_0    conda-forge    
dill                      0.3.5.1            pyhd8ed1ab_0    conda-forge    
fonttools                 4.37.1          py310he2412df_0    conda-forge    
freetype                  2.12.1               h546665d_0    conda-forge
gettext                   0.19.8.1          ha2e2712_1008    conda-forge
glib                      2.72.1               h7755175_0    conda-forge
glib-tools                2.72.1               h7755175_0    conda-forge
greenlet                  1.1.3           py310h8a704f9_0    conda-forge
gst-plugins-base          1.20.3               h001b923_1    conda-forge
gstreamer                 1.20.3               h6b5321d_1    conda-forge
icu                       70.1                 h0e60522_0    conda-forge
idna                      3.3                pyhd8ed1ab_0    conda-forge
importlib-metadata        4.11.4          py310h5588dad_0    conda-forge
importlib_metadata        4.11.4               hd8ed1ab_0    conda-forge
importlib_resources       5.9.0              pyhd8ed1ab_0    conda-forge
intel-openmp              2022.1.0          h57928b3_3787    conda-forge
isort                     5.10.1             pyhd8ed1ab_0    conda-forge
joblib                    1.1.0              pyhd8ed1ab_0    conda-forge
jpeg                      9e                   h8ffe710_2    conda-forge
kiwisolver                1.4.4           py310h476a331_0    conda-forge
krb5                      1.19.3               h1176d77_0    conda-forge
lazy-object-proxy         1.7.1           py310he2412df_1    conda-forge
lcms2                     2.12                 h2a16943_0    conda-forge
lerc                      4.0.0                h63175ca_0    conda-forge
libblas                   3.9.0              16_win64_mkl    conda-forge
libbrotlicommon           1.0.9                h8ffe710_7    conda-forge
libbrotlidec              1.0.9                h8ffe710_7    conda-forge
libbrotlienc              1.0.9                h8ffe710_7    conda-forge
libcblas                  3.9.0              16_win64_mkl    conda-forge
libclang                  14.0.6          default_h77d9078_0    conda-forge
libclang13                14.0.6          default_h77d9078_0    conda-forge
libdeflate                1.13                 h8ffe710_0    conda-forge
libffi                    3.4.2                h8ffe710_5    conda-forge
libglib                   2.72.1               h3be07f2_0    conda-forge
libiconv                  1.16                 he774522_0    conda-forge
liblapack                 3.9.0              16_win64_mkl    conda-forge
liblapacke                3.9.0              16_win64_mkl    conda-forge
libogg                    1.3.4                h8ffe710_1    conda-forge
libpng                    1.6.37               h1d00b33_4    conda-forge
libsqlite                 3.39.3               hcfcfb64_0    conda-forge
libtiff                   4.4.0                h92677e6_3    conda-forge
libuv                     1.44.2               h8ffe710_0    conda-forge
libvorbis                 1.3.7                h0e60522_0    conda-forge
libwebp-base              1.2.4                h8ffe710_0    conda-forge
libxcb                    1.13              hcd874cb_1004    conda-forge
libzlib                   1.2.12               h8ffe710_2    conda-forge
m2w64-gcc-libgfortran     5.3.0                         6    conda-forge
m2w64-gcc-libs            5.3.0                         7    conda-forge
m2w64-gcc-libs-core       5.3.0                         7    conda-forge
m2w64-gmp                 6.1.0                         2    conda-forge
m2w64-libwinpthread-git   5.0.0.4634.697f757               2    conda-forge
mako                      1.2.2              pyhd8ed1ab_0    conda-forge
markupsafe                2.1.1           py310he2412df_1    conda-forge
matplotlib                3.5.3           py310h5588dad_2    conda-forge
matplotlib-base           3.5.3           py310h7329aa0_2    conda-forge
mccabe                    0.7.0              pyhd8ed1ab_0    conda-forge
mkl                       2022.1.0           h6a75c08_874    conda-forge
mkl-devel                 2022.1.0           h57928b3_875    conda-forge
mkl-include               2022.1.0           h6a75c08_874    conda-forge
msys2-conda-epoch         20160418                      1    conda-forge
munkres                   1.1.4              pyh9f0ad1d_0    conda-forge
numpy                     1.23.2          py310h8a5b91a_0    conda-forge
openjpeg                  2.5.0                hc9384bd_1    conda-forge
openssl                   1.1.1q               h8ffe710_0    conda-forge
optuna                    3.0.1              pyhd8ed1ab_0    conda-forge
packaging                 21.3               pyhd8ed1ab_0    conda-forge
pandas                    1.4.4           py310h1c4a608_0    conda-forge
patsy                     0.5.2              pyhd8ed1ab_0    conda-forge
pbr                       5.10.0             pyhd8ed1ab_0    conda-forge
pcre                      8.45                 h0e60522_0    conda-forge
pillow                    9.2.0           py310h52929f7_2    conda-forge
pip                       22.2.2             pyhd8ed1ab_0    conda-forge
platformdirs              2.5.2              pyhd8ed1ab_1    conda-forge
ply                       3.11                       py_1    conda-forge
prettytable               3.4.1              pyhd8ed1ab_0    conda-forge
pthread-stubs             0.4               hcd874cb_1001    conda-forge
pycparser                 2.21               pyhd8ed1ab_0    conda-forge
pylint                    2.15.2             pyhd8ed1ab_0    conda-forge
pyopenssl                 22.0.0             pyhd8ed1ab_0    conda-forge
pyparsing                 3.0.9              pyhd8ed1ab_0    conda-forge
pyperclip                 1.8.2              pyhd8ed1ab_2    conda-forge
pyqt                      5.15.7          py310hbabf5d4_0    conda-forge
pyqt5-sip                 12.11.0         py310h8a704f9_0    conda-forge
pyreadline3               3.4.1           py310h5588dad_0    conda-forge
pysocks                   1.7.1              pyh0701188_6    conda-forge
python                    3.10.6          h9a09f29_0_cpython    conda-forge
python-dateutil           2.8.2              pyhd8ed1ab_0    conda-forge
python_abi                3.10                    2_cp310    conda-forge
pytorch                   1.12.1          py3.10_cuda11.3_cudnn8_0    pytorch
pytorch-mutex             1.0                        cuda    pytorch
pytz                      2022.2.1           pyhd8ed1ab_0    conda-forge
pyyaml                    6.0             py310he2412df_4    conda-forge
qt-main                   5.15.6               hf0cf448_0    conda-forge
requests                  2.28.1             pyhd8ed1ab_1    conda-forge
scikit-learn              1.1.2           py310h3a564e9_0    conda-forge
scipy                     1.8.1           py310h7c00807_2    conda-forge
setuptools                65.3.0             pyhd8ed1ab_1    conda-forge
sip                       6.6.2           py310h8a704f9_0    conda-forge
six                       1.16.0             pyh6c4a22f_0    conda-forge
sqlalchemy                1.4.41          py310h8d17308_0    conda-forge
statsmodels               0.13.2          py310h2873277_0    conda-forge
stevedore                 4.0.0              pyhd8ed1ab_0    conda-forge
tbb                       2021.5.0             h91493d7_2    conda-forge
threadpoolctl             3.1.0              pyh8a188c0_0    conda-forge
tk                        8.6.12               h8ffe710_0    conda-forge
toml                      0.10.2             pyhd8ed1ab_0    conda-forge
tomli                     2.0.1              pyhd8ed1ab_0    conda-forge
tomlkit                   0.11.4             pyha770c72_0    conda-forge
torchaudio                0.12.1              py310_cu113    pytorch
torchvision               0.13.1              py310_cu113    pytorch
tornado                   6.2             py310he2412df_0    conda-forge
tqdm                      4.64.1             pyhd8ed1ab_0    conda-forge
typing                    3.10.0.0           pyhd8ed1ab_0    conda-forge
typing_extensions         4.3.0              pyha770c72_0    conda-forge
tzdata                    2022c                h191b570_0    conda-forge
ucrt                      10.0.20348.0         h57928b3_0    conda-forge
unicodedata2              14.0.0          py310he2412df_1    conda-forge
urllib3                   1.26.11            pyhd8ed1ab_0    conda-forge
vc                        14.2                 hb210afc_7    conda-forge
vs2015_runtime            14.29.30139          h890b9b1_7    conda-forge
wcwidth                   0.2.5              pyh9f0ad1d_2    conda-forge
wheel                     0.37.1             pyhd8ed1ab_0    conda-forge
win_inet_pton             1.1.0           py310h5588dad_4    conda-forge
wrapt                     1.14.1          py310he2412df_0    conda-forge
xorg-libxau               1.0.9                hcd874cb_0    conda-forge
xorg-libxdmcp             1.1.3                hcd874cb_0    conda-forge
xz                        5.2.6                h8d14728_0    conda-forge
yaml                      0.2.5                h8ffe710_2    conda-forge
zipp                      3.8.1              pyhd8ed1ab_0    conda-forge
zstd                      1.5.2                h7755175_4    conda-forge
