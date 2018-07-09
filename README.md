# HeadHunter #
HeadHunter is an implementation of [HyperFace](https://arxiv.org/abs/1603.01249) on the PYNQ platform.

It benefits of the hardware acceleration provided by the FPGA present on SoC.

Goal of the project is to detect the face with orientation and landmark, and the gender of the people from an image or an USB camera.
More details are explained on the [project report](headhunter-report.pdf).

This software is released under the Apache V2 License, see [LICENSE.txt](LICENSE.txt).

## Build project ##

For all the below steps is used Vivado 2017.2.

### HLS Synthesis ###

For each CNN layer there is a dedicated directory ./HeadHunter/hls\_*layer_name* with its source file.
In each of them you can run synthesis by:
```
cd ./HeadHunter/hls_*layer_name*
vivado_hls -f run_hls.tcl
```

### Vivado Synthesis ###

For generate bitstream, go into ./HeadHunter/vivado and run for each layer:
```
cd ./HeadHunter/vivado
vivado -mode batch -source run_vivado_*layer_name*.tcl
```

### Pynq Setup ###

Copy the ./HeadHunter\_Pynq directory on Pynq.

It's required the **Dlib** library, version 19.1.0. You can install it with `pip3.6 install dlib==19.1.0`.
All the others dependencies are satisfied by Pynq framework itself.

You can run the model in different ways. With the root privileges you can run:

- `./HeadHunter -f *imageFile*`

  Run the model on the *imageFile* and write the output image in res\_*imageFile*
    
- `./HeadHunter`

  Run the model with the image taken from an USB camera.
  See the output as live\_LiveCam\_*iteration*.jpg
  
- `./HeadHunter.ipynb`

  Run the HeadHunter python notebook


## References ##

- [YouTube Video](https://www.youtube.com/watch?v=BngGK_HhvtQ) with an illustrated demo.
- [HyperFace](https://arxiv.org/abs/1603.01249), the implemented CNN based on AlexNet and Fusion.
- A python implementation used for reference can be found [here](https://github.com/takiyu/hyperface).

### Xilinx Open Hardware Details ###

Team number: xohw18-415

Project name: HeadHunter

Date: 30/6/2018

University name: Politecnico di Milano

Supervisor name: Marco Santambrogio

Supervisor e-mail: marco.santambrogio@polimi.it

Participant(s): Antonio Di Bello, Anna Maria Nestorov, Alberto Scolari

Email:antonio.dibello@mail.polimi.it , annamaria.nestorov@mail.polimi.it , alberto.scolari@polimi.it

Board used: PYNQ-Z1

Vivado Version: 2017.2

