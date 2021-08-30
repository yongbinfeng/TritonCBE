#!/usr/bin/python

# Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""
Testing script modified from https://github.com/triton-inference-server/server/blob/r21.02/qa/L0_backend_identity/identity_test.py
"""

import argparse
import numpy as np
import os
import re
import sys
import requests as httpreq
from builtins import range
import tritongrpcclient as grpcclient
import tritonhttpclient as httpclient
from tritonclientutils import np_to_triton_dtype

FLAGS = None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v',
                        '--verbose',
                        action="store_true",
                        required=False,
                        default=False,
                        help='Enable verbose output')
    parser.add_argument('-u',
                        '--url',
                        type=str,
                        required=False,
                        help='Inference server URL.')
    parser.add_argument(
        '-i',
        '--protocol',
        type=str,
        required=False,
        default='http',
        help='Protocol ("http"/"grpc") used to ' +
        'communicate with inference service. Default is "http".')

    FLAGS = parser.parse_args()
    if (FLAGS.protocol != "http") and (FLAGS.protocol != "grpc"):
        print("unexpected protocol \"{}\", expects \"http\" or \"grpc\"".format(
            FLAGS.protocol))
        exit(1)

    client_util = httpclient if FLAGS.protocol == "http" else grpcclient

    if FLAGS.url is None:
        FLAGS.url = "localhost:8020" if FLAGS.protocol == "http" else "localhost:8021"

    model_name = "p2r"
    request_parallelism = 1
    shape = [2, 2]
    with client_util.InferenceServerClient(FLAGS.url,
                                           concurrency=request_parallelism,
                                           verbose=FLAGS.verbose) as client:
        requests = []
        inputs = []
        for i in range(request_parallelism):
            input_atrk_par = np.array([-12.806846618652344, -7.723824977874756, 38.13014221191406,0.23732035065189902, -2.613372802734375, 0.35594117641448975],dtype=np.float32)
            input_atrk_cov = np.array([6.290299552347278e-07,4.1375109560704004e-08,7.526661534029699e-07,2.0973730840978533e-07,1.5431574240665213e-07,9.626245400795597e-08,-2.804026640189443e-06,6.219111130687595e-06,2.649119409845118e-07,0.00253512163402557,-2.419662877381737e-07,4.3124190760040646e-07,3.1068903991780678e-09,0.000923913115050627,0.00040678296006807003,-7.755406890332818e-07,1.68539375883925e-06,6.676875566525437e-08,0.0008420574605423793,7.356584799406111e-05,0.0002306247719158348], dtype=np.float32)
            input_atrk_q = np.array([1], dtype=np.int32)
            input_ahit_pos = np.array([-20.7824649810791, -12.24150276184082, 57.8067626953125], dtype=np.float32)
            input_ahit_cov = np.array([2.545517190810642e-06,-2.6680759219743777e-06,2.8030024168401724e-06,0.00014160551654640585,0.00012282167153898627,11.385087966918945], dtype=np.float32)
            inputs.append(
                client_util.InferInput("INPUT_ATRK_PAR", input_atrk_par.shape,
                                       np_to_triton_dtype(input_atrk_par.dtype))
            )
            inputs[-1].set_data_from_numpy(input_atrk_par)
            inputs.append(
                client_util.InferInput("INPUT_ATRK_COV", input_atrk_cov.shape,
                                       np_to_triton_dtype(input_atrk_cov.dtype))
            )
            inputs[-1].set_data_from_numpy(input_atrk_cov)
            inputs.append(
                client_util.InferInput("INPUT_ATRK_Q", input_atrk_q.shape,
                                       np_to_triton_dtype(input_atrk_q.dtype))
            )
            inputs[-1].set_data_from_numpy(input_atrk_q)
            inputs.append(
                client_util.InferInput("INPUT_AHIT_COV", input_ahit_cov.shape,
                                       np_to_triton_dtype(input_ahit_cov.dtype))
            )
            inputs[-1].set_data_from_numpy(input_ahit_cov)
            inputs.append(
                client_util.InferInput("INPUT_AHIT_POS", input_ahit_pos.shape,
                                       np_to_triton_dtype(input_ahit_pos.dtype))
            )
            inputs[-1].set_data_from_numpy(input_ahit_pos)
            requests.append(client.async_infer(model_name, inputs))

        for i in range(request_parallelism):
            # Get the result from the initiated asynchronous inference request.
            # Note the call will block till the server responds.
            results = requests[i].get_result()
            print(results)

            output_data = results.as_numpy("OUTPUT0")
            if output_data is None:
                print("error: expected 'OUTPUT0'")
                sys.exit(1)

            print("output data: ", output_data)


    print("Passed all tests!")
