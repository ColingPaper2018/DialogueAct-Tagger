from sklearn.base import BaseEstimator, TransformerMixin
from torchtext.data import Dataset, Example
from typing import List, Optional
from corpora.corpus import Utterance
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.base import TransformerMixin  # gives fit_transform method for free


class ItemSelector(BaseEstimator, TransformerMixin):
    """For data grouped by feature, select subset of data at a provided key.

    The data is expected to be stored in a 2D data structure, where the first
    index is over features and the second is over samples.  i.e.

    >> len(data[key]) == n_samples

    Please note that this is the opposite convention to scikit-learn feature
    matrixes (where the first index corresponds to sample).

    ItemSelector only requires that the collection implement getitem
    (data[key]).  Examples include: a dict of lists, 2D numpy array, Pandas
    DataFrame, numpy record array, etc.

    >> data = {'a': [1, 5, 2, 5, 2, 8],
               'b': [9, 4, 1, 4, 1, 3]}
    >> ds = ItemSelector(key='a')
    >> data['a'] == ds.transform(data)

    ItemSelector is not designed to handle data grouped by sample.  (e.g. a
    list of dicts).  If your data is structured this way, consider a
    transformer along the lines of `sklearn.feature_extraction.DictVectorizer`.

    Parameters
    ----------
    key : hashable, required
        The key corresponding to the desired value in a mappable.
    """

    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data):
        return [datum[self.key] for datum in data]


class DataFrameDataset(Dataset):
    """Class for using pandas DataFrames as a datasource"""

    def __init__(self, examples, fields, filter_pred=None):
        """
        Create a dataset from a pandas dataframe of examples and Fields
        Arguments:
            examples pd.DataFrame: DataFrame of examples
            fields {str: Field}: The Fields to use in this tuple. The
                string is a field name, and the Field is the associated field.
            filter_pred (callable or None): use only exanples for which
                filter_pred(example) is true, or use all examples if None.
                Default is None
        """
        self.examples = examples.apply(
            SeriesExample.fromSeries, args=(fields,), axis=1
        ).tolist()
        if filter_pred is not None:
            self.examples = filter(filter_pred, self.examples)
        self.fields = dict(fields)
        # Unpack field tuples
        for n, f in list(self.fields.items()):
            if isinstance(n, tuple):
                self.fields.update(zip(n, f))
                del self.fields[n]


class SeriesExample(Example):
    """Class to convert a pandas Series to an Example"""

    @classmethod
    def fromSeries(cls, data, fields):
        return cls.fromdict(data.to_dict(), fields)

    @classmethod
    def fromdict(cls, data, fields):
        ex = cls()

        for key, field in fields.items():
            if key not in data:
                raise ValueError(
                    "Specified key {} was not found in " "the input data".format(key)
                )
            if field is not None:
                setattr(ex, key, field.preprocess(data[key]))
            else:
                setattr(ex, key, data[key])
        return ex


def stringify_tags(
    dataset: List[Utterance],
    attribute: str,
    filter_attr: Optional[str] = None,
    filter_value: Optional[str] = None,
):
    stringified_dataset = []
    for utterance in dataset:
        new_tags = []
        new_context = []
        for tag in utterance.tags:
            if filter_value is None or getattr(tag, filter_attr).value == filter_value:
                new_tags.append(getattr(tag, attribute).value)
        for tag in utterance.context[0].tags:
            if filter_value is None or getattr(tag, filter_attr).value == filter_value:
                new_context.append(getattr(tag, attribute).value)
        if len(new_tags) > 0:
            stringified_dataset.append(
                Utterance(
                    speaker_id=utterance.speaker_id,
                    tags=new_tags,
                    context=new_context,
                    text=utterance.text,
                )
            )
    return stringified_dataset


class MyLabelBinarizer(TransformerMixin):
    def __init__(self, *args, **kwargs):
        self.encoder = MultiLabelBinarizer(*args, **kwargs)

    def fit(self, x, y=0):
        self.encoder.fit(x)
        return self

    def transform(self, x, y=0):
        return self.encoder.transform(x)
