# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
from __future__ import with_statement
"""
This module implements an object to deal with experimental paradigms.
In fMRI data analysis, there are two main types of experimental
paradigms: block and event-related paradigms. They correspond to 2
classes EventRelatedParadigm and BlockParadigm. Both are implemented
here, together with functions to write paradigms to csv files.

Notes
-----
Although the Paradigm object have no notion of session or acquisitions
(they are assumed to correspond to a sequential acquisition, called
'session' in SPM jargon), the .csv file used to represent paradigm may
be multi-session, so it is assumed that the first column of a file
yielding a paradigm is in fact a session index

Author: Bertrand Thirion, 2009-2011
"""

import numpy as np

from ...utils.compat3 import open4csv

##########################################################
# Paradigm handling
##########################################################


class Paradigm(object):
    """ Simple class to handle the experimental paradigm in one session
    """

    def __init__(self, con_id=None, onset=None, amplitude=None):
        """
        Parameters
        ----------
        con_id: array of shape (n_events), type = string, optional
               identifier of the events
        onset: array of shape (n_events), type = float, optional,
               onset time (in s.) of the events
        amplitude: array of shape (n_events), type = float, optional,
                   amplitude of the events (if applicable)
        """
        self.con_id = con_id
        self.onset = onset
        self.amplitude = amplitude
        if con_id is not None:
            self.n_events = len(con_id)
            try:
                # this is only for backward compatibility:
                #if con_id were integers, they become a string
                self.con_id = np.array(['c' + str(int(float(c)))
                                        for c in con_id])
            except:
                self.con_id = np.ravel(np.array(con_id)).astype('str')

        if onset is not None:
            if len(onset) != self.n_events:
                raise ValueError(
                    'inconsistent definition of ids and onsets')
            self.onset = np.ravel(np.array(onset)).astype(np.float)
        if amplitude is not None:
            if len(amplitude) != self.n_events:
                raise ValueError('inconsistent definition of amplitude')
            self.amplitude = np.ravel(np.array(amplitude))
        self.type = 'event'
        self.n_conditions = len(np.unique(self.con_id))

    def write_to_csv(self, csv_file, session='0'):
        """ Write the paradigm to a csv file

        Parameters
        ----------
        csv_file: string, path of the csv file
        session: string, optional, session identifier
        """
        import csv
        with open4csv(csv_file, "w") as fid:
            writer = csv.writer(fid, delimiter=' ')
            n_pres = np.size(self.con_id)
            sess = np.repeat(session, n_pres)
            pdata = np.vstack((sess, self.con_id, self.onset)).T

            # add the duration information
            if self.type == 'event':
                duration = np.zeros(np.size(self.con_id))
            else:
                duration = self.duration
            pdata = np.hstack((pdata, np.reshape(duration, (n_pres, 1))))

            # add the amplitude information
            if self.amplitude is not None:
                amplitude = np.reshape(self.amplitude, (n_pres, 1))
                pdata = np.hstack((pdata, amplitude))

            # write pdata
            for row in pdata:
                writer.writerow(row)


class EventRelatedParadigm(Paradigm):
    """ Class to handle event-related paradigms
    """

    def __init__(self, con_id=None, onset=None, amplitude=None):
        """
        Parameters
        ----------
        con_id: array of shape (n_events), type = string, optional
               id of the events (name of the experimental condition)
        onset: array of shape (n_events), type = float, optional
               onset time (in s.) of the events
        amplitude: array of shape (n_events), type = float, optional,
                   amplitude of the events (if applicable)
        """
        Paradigm.__init__(self, con_id, onset, amplitude)


class BlockParadigm(Paradigm):
    """ Class to handle block paradigms
    """

    def __init__(self, con_id=None, onset=None, duration=None, amplitude=None):
        """
        Parameters
        ----------
        con_id: array of shape (n_events), type = string, optional
               id of the events (name of the experimental condition)
        onset: array of shape (n_events), type = float, optional
               onset time (in s.) of the events
        amplitude: array of shape (n_events), type = float, optional,
                   amplitude of the events (if applicable)
        """
        Paradigm.__init__(self, con_id, onset, amplitude)
        self.duration = duration
        self.type = 'block'
        if duration is not None:
            if len(duration) != self.n_events:
                raise ValueError('inconsistent definition of duration')
            self.duration = np.ravel(np.array(duration))


def load_paradigm_from_csv_file(path, session=None):
    """
    Read a (.csv) paradigm file consisting of values yielding
    (occurrence time, (duration), event ID, modulation)
    and returns a paradigm instance or a dictionary of paradigm instances

    Parameters
    ----------
    path: string,
          path to a .csv file that describes the paradigm
    session: string, optional, session identifier
             by default the output is a dictionary
             of session-level dictionaries indexed by session

    Returns
    -------
    paradigm, paradigm instance (if session is provided), or
              dictionary of paradigm instances otherwise,
              the resulting session-by-session paradigm

    Notes
    -----
    It is assumed that the csv file contains the following columns:
    (session id, condition id, onset),
    plus possibly (duration) and/or (amplitude).
    If all the durations are 0, the paradigm will be handled as event-related.

    Fixme
    -----
    would be much clearer if amplitude was put before duration in the .csv
    """
    import csv
    with open4csv(path, 'r') as csvfile:
        dialect = csv.Sniffer().sniff(csvfile.read())
        csvfile.seek(0)
        reader = csv.reader(csvfile, dialect)

        # load the csv as a paradigm array
        sess, cid, onset, amplitude, duration = [], [], [], [], []
        for row in reader:
            sess.append(row[0])
            cid.append(row[1])
            onset.append(float(row[2]))
            if len(row) > 3:
                duration.append(float(row[3]))
            if len(row) > 4:
                amplitude.append(row[4])

        paradigm_info = [np.array(sess), np.array(cid), np.array(onset),
                    np.array(duration), np.array(amplitude)]
        paradigm_info = paradigm_info[:len(row)]

    def read_session(paradigm_info, session):
        """ return a paradigm instance corresponding to session
        """
        ps = (paradigm_info[0] == session)
        if np.sum(ps) == 0:
            return None
        ampli = np.ones(np.sum(ps))
        if len(paradigm_info) > 4:
            _, cid, onset, duration, ampli = [lp[ps] for lp in paradigm_info]
            if (duration == 0).all():
                paradigm = EventRelatedParadigm(cid, onset, ampli)
            else:
                paradigm = BlockParadigm(cid, onset, duration, ampli)
        elif len(paradigm_info) > 3:
            _, cid, onset, duration = [lp[ps] for lp in paradigm_info]
            paradigm = BlockParadigm(cid, onset, duration, ampli)
        else:
            _, cid, onset = [lp[ps] for lp in paradigm_info]
            paradigm = EventRelatedParadigm(cid, onset, ampli)
        return paradigm

    sessions = np.unique(paradigm_info[0])
    if session is None:
        paradigm = {}
        for session in sessions:
            paradigm[session] = read_session(paradigm_info, session)
    else:
        paradigm = read_session(paradigm_info, session)
    return paradigm
