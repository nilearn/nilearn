import numpy as np
from nilearn._utils.fixes import LabelBinarizer


def test_labelbinarizer_backport():
    np.testing.assert_array_equal(
        LabelBinarizer().fit_transform(np.arange(4)),
        np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]]))

    np.testing.assert_array_equal(
        LabelBinarizer(pos_label=1, neg_label=-1).fit_transform(np.arange(4)),
        np.array([[1, -1, -1, -1],
                  [-1, 1, -1, -1],
                  [-1, -1, 1, -1],
                  [-1, -1, -1, 1]]))


def test_labelbinarizer_misc_use_cases():
    lb = LabelBinarizer()
    np.testing.assert_array_equal(lb.fit([1, 2, 6, 4, 2]).classes_,
                                  [1, 2, 4, 6])
    np.testing.assert_array_equal(lb.transform([1, 6]),
                                  [[1, 0, 0, 0],
                                   [0, 0, 0, 1]])

    lb = LabelBinarizer()
    np.testing.assert_array_equal(lb.fit([[0, 1, 1], [1, 0, 0]]).classes_,
                                  [0, 1])

    clf = LabelBinarizer()
    np.testing.assert_array_equal(clf.fit([1, 2, 6, 4, 2]).classes_,
                                  [1, 2, 4, 6])
    np.testing.assert_array_equal(clf.transform([1, 6]),
                                  [[1., 0., 0., 0.],
                                   [0., 0., 0., 1.]])

    np.testing.assert_array_equal(clf.fit_transform([(1, 2), (3,)]),
                                  [[1., 1., 0.],
                                   [0., 0., 1.]])
    np.testing.assert_array_equal(clf.classes_, [1, 2, 3])
