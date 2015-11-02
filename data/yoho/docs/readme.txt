
		   README File for the YOHO Corpus

The YOHO Speaker Verification Corpus supports development, training
and testing of speaker verification systems that use limited
vocabulary, free-text input.  The particular vocabulary employed in
this collection consists of two-digit numbers ("thirty-four",
"sixty-one", etc), spoken continuously in sets of three (e.g.
"36-45-89").

This release of YOHO has been designed, with regard to the quantity
and organization of data, to answer the following question: does a
speaker verification system perform at 0.01% False-Accept and 0.1%
False-Reject at 75% confidence with a 50% probability of passing the
test?  There are 138 speakers (108 male, 30 female); for each speaker,
there are 4 enrollment sessions of 24 utterances each, and 10
verification sessions of four utterences each, for a total of 136
utterances in 14 sessions per speaker.

Description of waveform data
----------------------------

The sample rate for the speech files is 8 kHz, and the sample coding
is 12-bit linear (stored as 16-bit words).  For publication on CD-ROM,
each waveform file has been given an appropriate SPHERE file header,
and has been compressed using the "shorten" waveform compression
algorithm developed by Tony Robinson at Cambridge University.  The
SPHERE implementation of the "shorten" decompression process (see the
program "w_decode" in the SPHERE package) allows the output to be
written in either "big-endian" or "little-endian" format -- that is,
either high-byte or low-byte first in the 16-bit word.

Use and availability of SPHERE
------------------------------

The SPHERE software package, developed by the speech group at the
National Institute of Standards and Technology (NIST) was used to
incorporate the SPHERE headers and apply waveform compression.  It can
be used to access and modify header information, remove headers from
the files, and uncompress (and recompress) the waveform data.  For
convenience, we have included on this disc (in the "sphere" directory)
the most recent release of the SPHERE package as of the time of this
publication (version 2.1 Beta; this was the release used in preparing
the corpus for publication.)  Users are advised that NIST is
maintaining the package, and updating or correcting the software as
needed; when a more recent release becomes available, it can be
obtained for free directly from NIST, by means of anonymous ftp, via
the ftp host "jaguar.ncsl.nist.gov", in the "pub" directory.

Organization of the corpus
--------------------------

The corpus is divided into "enrollment" and "verification" segments
(in the "enroll" and "verify" diorectories"); each segment contains
data from all 138 speakers, who are designated by 3-digit numbers
(these are the directory names contained under "enroll" and "verify").
There are four enrollement sessions per speaker, numbered 1 through 4,
and each session contains 24 utterances; thus, each speaker directory
under "enroll" contains subdirectories "1", "2", "3" and "4", and each
of these contains 24 waveform files.  The name of each waveform file
indicates the "text" of the utterance in that file; for instance,
"62_31_53.wav" contains the phrase "sixty-two thirty-one fifty-three".
Each speaker directory under "verify" contains 10 sessions, numbered 1
through 10, and each session contains four utterances; again, the file
name for each utterance indicates the prompting text.

Other documentation
-------------------

The file "yoho_db.doc" contains portions of larger reports by Joseph
Campbell, Al Higgins, and others, that describe the creation of YOHO
in greater detail.

The file "speaker.doc" contains a list of the 138 speaker designations
(the 3-digit numbers), together with the speaker's gender ("M" or "F")
and, in most cases, the speaker's geographic origin.

