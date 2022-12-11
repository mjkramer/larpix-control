import time

import h5py
import numpy as np

from .hdf5format import dtypes, init_file

VERSION = "2.4"
DTYPE = dtypes[VERSION]["packets"]


def parse_msg(msg: np.array, io_group=0) -> np.array:
    assert msg[0] == ord("D")
    pacman_timestamp = msg[1] | (msg[2] << 8) | (msg[3] << 16) | (msg[4] << 24)
    nwords = msg[6] | (msg[7] << 8)

    npackets = nwords + 1
    packets = np.zeros((npackets,), dtype=DTYPE)

    for i in range(npackets):
        io_channel = 0
        chip_id = 0
        packet_type = 0
        downstream_marker = 0
        parity = 0
        valid_parity = 0
        channel_id = 0
        timestamp = 0
        dataword = 0
        trigger_type = 0
        local_fifo = 0
        shared_fifo = 0
        register_address = 0
        register_data = 0
        direction = 0
        local_fifo_events = 0
        shared_fifo_events = 0
        counter = 0
        fifo_diagnostics_enabled = 0
        first_packet = 0
        receipt_timestamp = 0

        if i == 0:
            packet_type = 4  # timestamp
            timestamp = pacman_timestamp
        else:
            wordstart = 8 + (16 * (i - 1))
            word = msg[wordstart : (wordstart + 16)]
            header, data = word[:8], word[8:]

            wordtype = header[0]

            if wordtype == 0x44:  # 'D'
                packet_type = 0  # data
                io_channel = header[1]
                receipt_timestamp = (
                    header[2] | (header[3] << 8) | (header[4] << 16) | (header[5] << 24)
                )
                chip_id = (data[0] >> 2) | (data[1] << 6)
                channel_id = data[1] >> 2
                timestamp = (
                    data[2]
                    | (data[3] << 8)
                    | (data[4] << 16)
                    | ((data[5] & 0x7F) << 24)
                )
                first_packet = (data[5] >> 7) & 1
                dataword = data[6]
                trigger_type = data[7] & 0x03
                local_fifo = (data[7] >> 2) & 0x03
                shared_fifo = (data[7] >> 4) & 0x03
                downstream_marker = (data[7] >> 6) & 1
                parity = (data[7] >> 7) & 1
                valid_parity = np.unpackbits(data).sum() % 2

                register_address = (data[1] >> 2) | (data[2] << 6)
                register_data = (data[2] >> 2) | (data[3] << 6)

            elif wordtype == 0x53:  # 'S'
                packet_type = 6  # sync
                trigger_type = sync_type = header[1]
                dataword = clk_source = header[2] & 0x01
                timestamp = (
                    header[4] | (header[5] << 8) | (header[6] << 16) | (header[7] << 24)
                )

            elif wordtype == 0x54:  # 'T'
                packet_type = 7  # trigger
                trigger_type = header[1]
                timestamp = (
                    header[4] | (header[5] << 8) | (header[6] << 16) | (header[7] << 24)
                )

        packets[i] = (
            io_group,
            io_channel,
            chip_id,
            packet_type,
            downstream_marker,
            parity,
            valid_parity,
            channel_id,
            timestamp,
            dataword,
            trigger_type,
            local_fifo,
            shared_fifo,
            register_address,
            register_data,
            direction,
            local_fifo_events,
            shared_fifo_events,
            counter,
            fifo_diagnostics_enabled,
            first_packet,
            receipt_timestamp,
        )

    return packets


def to_file_quick(filename, msg_list=[], io_groups=[], chip_list=[], mode="a"):
    with h5py.File(filename, mode) as f:
        init_file(f, VERSION, chip_list)

        packet_dset_name = "packets"
        if packet_dset_name not in f.keys():
            packet_dset = f.create_dataset(
                packet_dset_name,
                shape=(0,),
                maxshape=(None,),
                dtype=DTYPE,
            )
            start_index = 0
        else:
            packet_dset = f[packet_dset_name]
            start_index = packet_dset.shape[0]

        for msg, io_group in zip(msg_list, io_groups):
            packets = parse_msg(msg, io_group)
            packet_dset.resize(start_index + len(packets), axis=0)
            packet_dset[start_index:] = packets
            start_index += len(packets)
