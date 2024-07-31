# Play C++ in Jupyter Notebook

- [jupyter-xeus/xeus-cling: Jupyter kernel for the C++ programming language](https://github.com/jupyter-xeus/xeus-cling)

Highly recommend first try in [Binder - GitHub: jupyter-xeus/xeus-cling/stable](https://mybinder.org/v2/gh/jupyter-xeus/xeus-cling/stable?filepath=notebooks/xcpp.ipynb)

- [Using third-party libraries · Issue #87 · jupyter-xeus/xeus-cling](https://github.com/jupyter-xeus/xeus-cling/issues/87#issuecomment-349053121)
- [Build options — xeus-cling documentation](https://xeus-cling.readthedocs.io/en/latest/build_options.html#using-third-party-libraries)

```cpp
// If you want to add include path
#pragma cling add_include_path("inc_directory")

// If you want to add library path
#pragma cling add_library_path("lib_directory")

// If you want to load library
#pragma cling load("libname")

You can use all this commands in a code cell in Jupyter notebook.
```

---

## Docker

[**`docker/run_xeus-cling_docker.sh`**](docker/run_xeus-cling_docker.sh)

> - [sehrig/cling - Docker Image | Docker Hub](https://hub.docker.com/r/sehrig/cling)
>
> ```bash
> docker run -it -p 8888:8888 sehrig/cling jupyter-notebook
> ```
>
> - [xeus-cling C++ Jupyter kernel inside a docker container](https://gist.github.com/dsuess/059b86ea55d639bb99175c9a8cd2ca3e)

## Install without conda/mamba

- [Install without Anaconda/Miniconda · Issue #327 · jupyter-xeus/xeus-cling](https://github.com/jupyter-xeus/xeus-cling/issues/327)
- [install xeus-cling using pip? · Issue #301 · jupyter-xeus/xeus-cling](https://github.com/jupyter-xeus/xeus-cling/issues/301)
- [pojntfx/xeus-cling-binaries: Weekly builds of https://github.com/jupyter-xeus/xeus-cling.](https://github.com/pojntfx/xeus-cling-binaries)

```bash
# Fetch the xeus-cling binary package for your architecture (x86_64 and aarch64 are supported)
curl -L -o /tmp/xeus-cling.tar.gz https://github.com/pojntfx/xeus-cling-binaries/releases/download/latest/xeus-cling.$(uname -m).tar.gz

# Extract the package to /usr/local/xeus-cling. You must install in this prefix.
XEUS_PREFIX=/usr/local/xeus-cling
sudo mkdir -p ${XEUS_PREFIX}
sudo tar -C ${XEUS_PREFIX} -xzf /tmp/xeus-cling.tar.gz

# Install the kernels
sudo jupyter kernelspec install ${XEUS_PREFIX}/share/jupyter/kernels/xcpp11 --sys-prefix
sudo jupyter kernelspec install ${XEUS_PREFIX}/share/jupyter/kernels/xcpp14 --sys-prefix
sudo jupyter kernelspec install ${XEUS_PREFIX}/share/jupyter/kernels/xcpp17 --sys-prefix
```
