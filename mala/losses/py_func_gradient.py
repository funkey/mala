import tensorflow as tf

def py_func_gradient(func, inp, Tout, stateful=True, name=None, gradient_op=None):

    pyfunc_name = 'PyFuncGrad' + str(name)

    tf.RegisterGradient(pyfunc_name)(gradient_op)
    g = tf.get_default_graph()

    with g.gradient_override_map({
        "PyFunc": pyfunc_name,
        "PyFuncStateless": pyfunc_name}):

        return tf.py_func(func, inp, Tout, stateful=stateful, name=name)
