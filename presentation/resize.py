import numpy
import cv2


def bilinear_interpolation(nearest_colors, point_cord):
    point_color = [0, 0, 0]

    right_color = nearest_colors[0][0] * (1 - point_cord[0]) + nearest_colors[1][0] * point_cord[0]
    left_color = nearest_colors[0][1] * (1 - point_cord[0]) + nearest_colors[1][1] * point_cord[0]

    point_color = right_color * (1 - point_cord[1]) + left_color * point_cord[1]
    point_color = numpy.round(point_color).astype(numpy.uint8)

    return point_color


def resize(image, size, method='bilinear'):

    interpolation = {
        'bilinear': bilinear_interpolation
    }

    new_image = numpy.zeros([size[0], size[1], 3])

    def dimension_relation(old_dim, new_dim):
        a = [[-0.5, 1], [new_dim, 1]]
        b = [-0.5, old_dim + 0.5]
        cord_relation = numpy.linalg.solve(a, b)
        return cord_relation

    x_cord_coef = dimension_relation(image.shape[1], size[1])
    y_cord_coef = dimension_relation(image.shape[0], size[0])

    for row in range(size[0]):
        for column in range(size[1]):
            point_column = numpy.dot(x_cord_coef, numpy.array([column, 1]))
            point_row = numpy.dot(y_cord_coef, numpy.array([row, 1]))

            point_column = numpy.clip(point_column, 0, image.shape[1] - 1)
            point_row = numpy.clip(point_row, 0, image.shape[0] - 1)

            top = numpy.floor(point_row).astype(numpy.uint)
            right = numpy.ceil(point_column).astype(numpy.uint)
            bottom = numpy.ceil(point_row).astype(numpy.uint)
            left = numpy.floor(point_column).astype(numpy.uint)

            nearest_colors = [
                [image[top][left], image[top][right]],
                [image[bottom][left], image[bottom][right]]
            ]

            point_cord = [numpy.modf(point_row)[0], numpy.modf(point_column)[0]]

            new_color = interpolation.get(method)(nearest_colors, point_cord)

            new_image[row][column][:] = new_color

    return new_image.astype(numpy.uint8)


if __name__ == "__main__":
    image = cv2.imread("image/Lenna.png")
    # cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # new_image = resize(image, (1024, 1024), 'bilinear')
    # cv2.imshow("Frame", new_image)
    # cv2.waitKey(0)

    new_image = cv2.resize(image, (128, 128), interpolation=cv2.INTER_LINEAR)
    cv2.imshow("Frame", new_image)

