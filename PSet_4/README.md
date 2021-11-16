## Perceptron Implementation

Below is a visualization of the result of my final decision boundary when implementing the perceptron algorithm on a given set of data points.

![perceptron](https://user-images.githubusercontent.com/88719947/141918732-214fcba7-32c1-4014-be82-94d662235b6c.png)

## Linear Regression Implementation

I tried to implement an α = 0.07 and a limit of 500. While I was implementing my linear regression algorithm with the different α values, I also calculated their corresponding costs. I noticed that really small values (i.e α=0.0001, 0.005, and 0.01) never had a chance to truly find the minimum because there were not enough iterations. Values greater than 1 tend to be too large of a learning rate and had much larger errors. α=5 and α=10 did not even have a plane that showed up within the bounds of the graph. As a result, I tried for a fairly small learning rate (α=0.07) with a much higher limit (limit=500). I wanted to see if this would result in a better convergence. These parameters did not do better than α=0.1, 0.5, or 1 with limits=100. I believe all of them found the local minimum.
