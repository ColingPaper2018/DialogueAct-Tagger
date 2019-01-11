import csv


class Corpus:
    def __init__(self, corpus_folder):
        self.csv_corpus = []
        pass

    def load_csv(self):
        raise NotImplementedError()

    def create_csv(self, dialogs):
        raise NotImplementedError()

    def dump_csv(self, out_file=None):
        if out_file is not None:
            with open(out_file, 'w') as out:
                csv_out = csv.writer(out)
                for row in self.csv_corpus:
                    csv_out.writerow(row)

    def get_corpus_name(self):
        raise NotImplementedError()

    def dump_iso_dimension_task_csv(self, out_file=None):
        return_csv = []
        for row in self.csv_corpus:
            new_utterance = self.corpus_tuple_to_iso_task_dimension(row)
            if new_utterance is not None:
                return_csv.append(new_utterance)
        if out_file is not None:
            with open(out_file, 'wb') as out:
                csv_out = csv.writer(out)
                for row in return_csv:
                    csv_out.writerow(row)
        return return_csv

    def dump_iso_dimension_som_csv(self, out_file=None):
        return_csv = []
        for row in self.csv_corpus:
            new_utterance = self.corpus_tuple_to_iso_som_dimension(row)
            if new_utterance is not None:
                return_csv.append(new_utterance)
        if out_file is not None:
            with open(out_file, 'wb') as out:
                csv_out = csv.writer(out)
                for row in return_csv:
                    csv_out.writerow(row)
        return return_csv

    def dump_iso_dimension_fb_csv(self, out_file=None):
        return_csv = []
        for row in self.csv_corpus:
            new_utterance = self.corpus_tuple_to_iso_fb_dimension(row)
            if new_utterance is not None:
                return_csv.append(new_utterance)
        if out_file is not None:
            with open(out_file, 'wb') as out:
                csv_out = csv.writer(out)
                for row in return_csv:
                    csv_out.writerow(row)
        return return_csv

    def dump_iso_task_csv(self, out_file=None):
        return_csv = []
        for row in self.csv_corpus:
            new_utterance = self.corpus_tuple_to_iso_task(row)
            if new_utterance is not None:
                return_csv.append(new_utterance)
        if out_file is not None:
            with open(out_file, 'wb') as out:
                csv_out = csv.writer(out)
                for row in return_csv:
                    csv_out.writerow(row)
        return return_csv

    def dump_iso_som_csv(self, out_file=None):
        return_csv = []
        for row in self.csv_corpus:
            new_utterance = self.corpus_tuple_to_iso_som(row)
            if new_utterance is not None:
                return_csv.append(new_utterance)
        if out_file is not None:
            with open(out_file, 'wb') as out:
                csv_out = csv.writer(out)
                for row in return_csv:
                    csv_out.writerow(row)
        return return_csv

    def dump_iso_fb_csv(self, out_file=None):
        return_csv = []
        for row in self.csv_corpus:
            new_utterance = self.corpus_tuple_to_iso_fb(row)
            if new_utterance is not None:
                return_csv.append(new_utterance)
        if out_file is not None:
            with open(out_file, 'wb') as out:
                csv_out = csv.writer(out)
                for row in return_csv:
                    csv_out.writerow(row)
        return return_csv

    @staticmethod
    def da_to_dimension(corpus_tuple):
        raise NotImplementedError()

    @staticmethod
    def da_to_cf(corpus_tuple):
        raise NotImplementedError()

    def corpus_tuple_to_iso_task_dimension(self, corpus_tuple):
        da = self.da_to_dimension(corpus_tuple)
        prevDA = self.da_to_cf((None, corpus_tuple[2], None, None, corpus_tuple[5], None))
        if prevDA is None:
            prevDA = "Other"
        if da is not None:
            if da != "Task":
                da = "Other"
            return tuple([corpus_tuple[0]] + [da, prevDA] + list(corpus_tuple[2:]))
        else:
            return None

    def corpus_tuple_to_iso_som_dimension(self, corpus_tuple):
        da = self.da_to_dimension(corpus_tuple)
        prevDA = self.da_to_cf((None, corpus_tuple[2], None, None, corpus_tuple[5], None))
        if prevDA is None:
            prevDA = "Other"
        if da is not None:
            if da != "SocialObligationManagement":
                da = "Other"
            return tuple([corpus_tuple[0]] + [da, prevDA] + list(corpus_tuple[2:]))
        else:
            return None

    def corpus_tuple_to_iso_fb_dimension(self, corpus_tuple):
        da = self.da_to_dimension(corpus_tuple)
        prevDA = self.da_to_cf((None, corpus_tuple[2], None, None, corpus_tuple[5], None))
        if prevDA is None:
            prevDA = "Other"
        if da is not None:
            if da != "Feedback":
                da = "Other"
            return tuple([corpus_tuple[0]] + [da, prevDA] + list(corpus_tuple[2:]))
        else:
            return None

    def corpus_tuple_to_iso_task(self, corpus_tuple):
        if self.da_to_dimension(corpus_tuple) == "Task":
            da = self.da_to_cf(corpus_tuple)
            prevDA = self.da_to_cf((None, corpus_tuple[2], None, None, corpus_tuple[5], None))
            if prevDA is None:
                prevDA = "Other"
            if da is not None:
                return tuple([corpus_tuple[0]] + [da, prevDA] + list(corpus_tuple[2:]))
            else:
                return None

    def corpus_tuple_to_iso_som(self, corpus_tuple):
        if self.da_to_dimension(corpus_tuple) == "SocialObligationManagement":
            da = self.da_to_cf(corpus_tuple)
            prevDA = self.da_to_cf((None, corpus_tuple[2], None, None, corpus_tuple[5], None))
            if prevDA is None:
                prevDA = "Other"
            if da is not None:
                return tuple([corpus_tuple[0]] + [da, prevDA] + list(corpus_tuple[2:]))
            else:
                return None

    def corpus_tuple_to_iso_fb(self, corpus_tuple):
        if self.da_to_dimension(corpus_tuple) == "Feedback":
            da = self.da_to_cf(corpus_tuple)
            prevDA = self.da_to_cf((None, corpus_tuple[2], None, None, corpus_tuple[5], None))
            if prevDA is None:
                prevDA = "Other"
            if da is not None:
                return tuple([corpus_tuple[0]] + [da, prevDA] + list(corpus_tuple[2:]))
            else:
                return None

    def dataload(self, tokenizer, multilabel=False):
        segment = "-1"
        x = []
        y = []
        for datapoint in self.csv_corpus:
            if datapoint[3] != segment and segment != "-1" and len(x) > 0:
                yield (x, y)
                x = []
                y = []
            segment = datapoint[3]
            dimension, cf = self.da_to_dimension(datapoint), self.da_to_cf(datapoint)
            if dimension is None or cf is None:
                x = []
                y = []
            else:
                tokenized = tokenizer(datapoint[0])
                x.extend([tok for tok in tokenized])
                if multilabel:
                    y.extend([[(dimension, cf)] for tok in tokenized])
                else:
                    y = dimension+cf


