import popsynth


def test_log_settings():

    popsynth.silence_warnings()
    popsynth.activate_warnings()

    popsynth.silence_logs()

    popsynth.activate_logs()

    popsynth.show_progress_bars()

    popsynth.silence_progress_bars()

    popsynth.loud_mode()

    popsynth.quiet_mode()
