#!/usr/bin/env python

import xbob.db.verification.filelist

# 0/ The database to use
name = 'mobile0-female'
db = xbob.db.verification.filelist.Database('protocols/mobio/mobile0-female/')
protocol = None

wav_input_dir = "/idiap/resource/database/mobio/denoisedAUDIO_16k/"
wav_input_ext = ".wav"

