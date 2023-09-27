$CMAKE --build $BASE_PATH/parflow/build
sudo $CMAKE --install $BASE_PATH/parflow/build --prefix /opt/parflow
python3 -m pip install /home/ubuntu/parflow/build/pftools/python
cp -R $BASE_PATH/parflow/build/pftools/python/parflow/tools/ref ~/.local/lib/python3.8/site-packages/parflow/tools/ref