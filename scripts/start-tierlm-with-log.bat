@echo off
rem tierlm with a detailed log, for tracking down a problem: the same as start-tierlm.bat, with more detail in the
rem window and everything it shows saved to a new file in the logs folder. Send that file in with a problem report.
rem (It names your PC's parts and the models you load, not what you write in chats.)
set "DETAILED=yes"
set "SAVE_LOG=yes"
call "%~dp0start-tierlm.bat" %*
