import pytest
import os

import elfi
import elfi.examples as ee


def test_bdm(recwarn):
    """Currently only works in unix-like systems and with a cloned repository"""
    cpp_path = ee.bdm.get_sources_path()

    do_cleanup = False
    if not os.path.isfile(cpp_path + '/bdm'):
        os.system('make -C {}'.format(cpp_path))
        do_cleanup = True

    assert os.path.isfile(cpp_path + '/bdm')

    with pytest.warns(RuntimeWarning):
        bdm = ee.bdm.get_model()

    # Copy the file here to run the test
    os.system('cp {}/bdm .'.format(cpp_path))

    # Should no longer warn
    bdm = ee.bdm.get_model()

    # Test that you can run the inference

    rej = elfi.Rejection(bdm, 'd', batch_size=100)
    rej.sample(20)

    # TODO: test the correctness of the result

    os.system('rm ./bdm')
    if do_cleanup:
        os.system('rm {}/bdm'.format(cpp_path))

