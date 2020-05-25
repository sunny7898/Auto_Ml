#%%
import smtplib, ssl

port = 465  # For SSL
smtp_server = "smtp.gmail.com"
sender_email = "sungoswami07@gmail.com"  # Enter your address
receiver_email = "sunnygoswami7898@gmail.com"  # Enter receiver address
password = "Your Password"
message = """\
Subject: Hi there

Best model has been created"""

context = ssl.create_default_context()
with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
    server.login(sender_email, password)
    server.sendmail(sender_email, receiver_email, message)

# %%
