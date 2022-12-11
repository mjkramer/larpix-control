import h5py
import numpy as np

from .hdf5format import dtypes

DTYPE = dtypes["2.4"]["packets"]

TEST_MSG1 = b"D\xdbA\x8bc\x00\x02\x00D\x1c\x9a\xa6@\x00\x00\x00,1\xbeK\x19\x80<LD\x1c&\xa7@\x00\x00\x00,1JL\x19\x80;\xcc"
TEST_MSG2 = b"D\xdbA\x8bc\x00\x01\x00D\x07\x8b\xa9@\x00\x00\x00\xf8\xcc\xca\xa7@\x80'@"


def parse_msg(msg, io_group=1):
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

            if wordtype == ord("D"):  # or TX?
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
                valid_parity = 1  # XXX

                register_address = (data[1] >> 2) | (data[2] << 6)
                register_data = (data[2] >> 2) | (data[3] << 6)

            elif wordtype == ord("S"):
                packet_type = 6  # sync
                trigger_type = sync_type = header[1]
                dataword = clk_source = header[2] & 0x01
                timestamp = (
                    header[4] | (header[5] << 8) | (header[6] << 16) | (header[7] << 24)
                )

            elif wordtype == ord("T"):
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


def to_file(filename):
    with h5py.File(filename, "w") as f:
        packet_dset = f.create_dataset(
            "packets", shape=(20,), maxshape=(None,), dtype=dtypes["2.4"]["packets"]
        )
        # packet_dset.resize
