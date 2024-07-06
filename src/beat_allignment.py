    def beat_alignment(
        self, waveform_1: torch.Tensor, waveform_2: torch.Tensor, sample_rate: int
    ):
        """Assumes sample rates are equal already."""
        HOP_LENGTH = 512

        tempo_1, beats_1 = librosa.beat.beat_track(
            y=waveform_1.numpy()[0],
            sr=sample_rate,
            tightness=80,
            trim=True,
            sparse=True,
            hop_length=HOP_LENGTH,
        )
        tempo_2, beats_2 = librosa.beat.beat_track(
            y=waveform_2.numpy()[0],
            sr=sample_rate,
            trim=True,
            sparse=True,
            hop_length=HOP_LENGTH,
        )

        fade = Fade(fade_in_len=0, fade_out_len=4410, fade_shape="exponential").to(
            waveform_1.device
        )

        print(f"Tempo 1: {tempo_1}, Tempo 2: {tempo_2}")
        print(f"Beats 1: {beats_1.shape}, Beats 2: {beats_2.shape}")

        # Handle cases where no beats are detected
        if len(beats_1) == 0 or len(beats_2) == 0:
            return waveform_2  # Return the original waveform if no beats are found

        # Each row of the `warping_path` array contains an index pair (n, m).
        dtw_cost_matrix, dtw_warping_path = librosa.sequence.dtw(
            beats_1, beats_2, subseq=True, metric="euclidean"
        )

        waveform_2_sync = np.zeros((waveform_2.shape[0], waveform_1.shape[1]))

        # dtw_cost_matrix, dtw_warping_path = librosa.sequence.dtw(
        #     beats_1, beats_2, subseq=True, metric="euclidean"
        # )

        print(f"DTW Distance: {dtw_cost_matrix[-1, -1]}")
        print(f"DTW Path: {dtw_warping_path}")

        # Align the beats
        for i in range(len(dtw_warping_path) - 2):
            if i > beats_1.shape[0] - 1 or i > beats_2.shape[0] - 1:
                break
            start_sample_1 = beats_1[dtw_warping_path[i, 0]] * HOP_LENGTH

            try:
                end_sample_1 = beats_1[dtw_warping_path[i, 0] + 1] * HOP_LENGTH
            except IndexError:
                print(
                    f"Tried to access {dtw_warping_path[i, 0] + 1} in {beats_1.shape[0]}"
                )
                end_sample_1 = waveform_1.shape[1]
            start_sample_2 = beats_2[dtw_warping_path[i, 1]] * HOP_LENGTH

            try:
                end_sample_2 = beats_2[dtw_warping_path[i, 1] + 1] * HOP_LENGTH
            except IndexError:
                print(
                    f"Tried to access {dtw_warping_path[i, 1] + 1} in {beats_2.shape[0]}"
                )
                end_sample_2 = waveform_2.shape[1]

            segment_2 = waveform_2[:, start_sample_2:end_sample_2]

            # Calculate stretch ratio based on sample durations
            rate = (end_sample_1 - start_sample_1) / (end_sample_2 - start_sample_2)

            # Time-stretch or shrink the segment
            stretched_segment = librosa.effects.time_stretch(
                y=segment_2.numpy(), rate=rate
            )

            # Pad or truncate the stretched segment to fit
            target_length = end_sample_1 - start_sample_1
            if stretched_segment.shape[1] < target_length:
                padding = np.zeros(
                    (
                        stretched_segment.shape[0],
                        target_length - stretched_segment.shape[1],
                    )
                )
                stretched_segment = np.hstack([stretched_segment, padding])
            elif stretched_segment.shape[1] > target_length:
                stretched_segment = stretched_segment[:, :target_length]

            # Insert the stretched/shrunk segment into the synchronized waveform
            # Use torchaudio.transforms.Fade
            stretched_segment = fade(torch.tensor(stretched_segment, device=waveform_1.device))
            waveform_2_sync[:, start_sample_1:end_sample_1] = stretched_segment.numpy()

        # If the resulting waveform is shorter, crop the other to match
        if waveform_2_sync.shape[1] < waveform_1.shape[1]:
            padding = np.zeros(
                (waveform_2_sync.shape[0], waveform_1.shape[1] - waveform_2_sync.shape[1])
            )
            waveform_2_sync = np.hstack([waveform_2_sync, padding])
        elif waveform_2_sync.shape[1] > waveform_1.shape[1]:
            waveform_2_sync = waveform_2_sync[:, : waveform_1.shape[1]]

        ret = torch.tensor(waveform_2_sync, device=waveform_1.device)
        print(f"Original Waveform 1: {waveform_1.shape}")
        print(f"Aligned Waveform 2: {ret.shape}")
        print(f"Original Waveform 2: {waveform_2.shape} ")

        return ret