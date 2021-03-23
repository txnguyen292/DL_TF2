import tensorflow as tf

A, B =tf.constant(3.0), tf.constant(6.0)
X = tf.Variable(20.0)

loss = tf.math.abs(A * X - B)

def train_step():
    with tf.GradientTape() as tape:
        loss = tf.math.abs(A * X - B)
    dX = tape.gradient(loss, X)

    print(f"X = {X.numpy():.2f}, dX = {dX:.2f}")
    X.assign(X - dX)
    return X

if __name__ == "__main__":
    for i in range(7):
        train_step()