FULL_THREE_PART_FEWSHOT_TEMPLATE = """I will show you a press release and ask you for some suggestions for a reporter to follow.
You will answer the following questions. Summarize each point you make with a separate sentence.
1. What parts of the press release are "spun", or written to present facts about the company in a positive light, and how might a critical news article de-spin these?
2. What are potential controversies, or possible story angles I can investigate in this press release?
3. What are some sources I should speak to, or resources I should use to pursue these angles? 

Answer each question seperately. List each part, controversy or source in a separate sentence. Don't list them together. Respond with a JSON, with the keys being "1", "2", "3" for each question, and the values being a list. Don't include any other formatting.

Here are some examples, with abbreviated press releases:

Example 1:
PRESS RELEASE: ```As we have stated many times in the past, Live Nation takes its responsibilities under the antitrust laws seriously and does not engage in behaviors that could justify antitrust litigation, let alone orders that would require it to alter fundamental business practices.```
```{{'1': ['The journalist can take a critical angle on the Live Nation Entertainment press release, focusing on the negative aspects of the Ticketmaster-Live Nation merger, including allegations of monopolistic behavior and its impact on ticket prices and market competition.'],
   '2': ['The journalist can challenge the press release by highlighting the negative consequences of the Ticketmaster-Live Nation merger, such as high ticket prices and low competition.',
    "The journalist can contrast the press release's claims of competitive behavior and innovation with allegations of a dominant market position leading to less choice and higher ticket fees.",
    'The journalist can emphasize the outrage and disappointment of fans, to illustrate the real-world impact of the issues.',
    "The journalist can question the effectiveness and sincerity of Live Nation's commitment to antitrust laws and consumer-friendly practices."],
   '3': ['An antitrust expert who testified against the Ticketmaster-Live Nation merger, offering insights into the potential priorities of the DOJ regarding antitrust enforcement.',
    'Customers or fans.',
    'Public lawmakers, including specific members of Congress who have publicly criticized the merger and called for action against the perceived monopoly.',
    "Comparative financial and operational data pre and post-merger, to illustrate the changes in the market dynamics and Live Nation's business performance."]}}
```

Example 2:
PRESS RELEASE: ```STAMFORD, Conn - Charter Communications, Inc. (along with its subsidiaries, the "Company" or "Charter") today provided an update on its contract negotiation dispute with The Walt Disney Company. Overview We respect the quality video products that The Walt Disney Company produces as well as the experience of its management team. But the current video ecosystem is broken, and we know there is a better path that will deliver video products with the choice consumers want.```
```{{'1': ['The journalist can present the ongoing dispute between Charter Communications and The Walt Disney Company, highlighting the impact on consumers.'],
   '2': ["The article can challenge Charter Communications' portrayal of the dispute by highlighting Disney's actions.",
    "It can contrast Charter's claims by noting Disney's efforts to transition its business models and streamlining its costs.",
    "It can provide a counter-narrative to Charter's assertion of trying to protect consumers from high costs by mentioning Charter's willingness to accept Disney's rate increase under certain conditions.",
    "The report implies that the dispute's impact on consumers extends beyond just costs, touching on how it affects access to content."],
   '3': ['Financial analysis might provide an external perspective on the financial implications of the dispute.',
    "Statements from a Charter Communications spokesperson to represent the company's stance directly.",
    "A blog post from Disney to customers, revealing Disney's side of the story.",
    'Another news source offering a credible journalistic perspective on the issue.']}}
```

Example 3:
PRESS RELEASE: ```Tesla cars come standard with advanced hardware capable of providing Autopilot features, and full self-driving capabilities--through software updates designed to improve functionality over time.```
```{{'1': ["The news article can explore controversy surrounding Tesla's marketing of its 'Full Self-Driving' and 'Autopilot' features, inclusing Tesla using misleading language and the potential consequences for Tesla."],
   '2': ["The article can challenge Tesla's claims by pointing out the discrepancy between Tesla's marketing language and the actual functionality of the 'Autopilot' and 'Full Self-Driving' features, which require active driver supervision.",
    'It can emphasize the seriousness of the misleading marketing by citing the potential legal repercussions Tesla faces, including the possibility of being banned from selling cars in California.',
    "It can highlight skepticism towards Tesla's claims by discussing past scrutiny from national safety organizations and the implications of the company's self-driving technology not being fully autonomous as advertised."],
   '3': 'The article can incorporate statements and actions regulatory agencies to frame the scrutiny and potential legal challenges Tesla faces.',
    "Comments and actions by the National Transportation Safety Board (NTSB) are included to discuss past concerns regarding Tesla's feature naming and implications.",
    "The journalist can look at Tesla's website to see if there is a disclaimer, indicating Tesla's acknowledgment of the limitations of Autopilot and Full Self-Driving features which contradicts their descriptions.",
    "An interview with Elon Musk can illustrate the importance placed on self-driving technology by Tesla's CEO.",
    'A DMV spokesperson can be quoted to suggest potential softer regulatory outcomes despite the serious allegations.'}}
```

Ok, now it's your turn. Again, you will see a press release and answer the following questions. Summarize each point you make with a separate sentence.
1. What parts of the press release are "spun", or written to present facts about the company in a positive light, and how might a critical news article de-spin these?
2. What are potential controversies, or possible story angles I can investigate in this press release?
3. What are some sources I should speak to, or resources I should use to pursue these angles? 

Answer each question seperately. List each part, controversy or source in a separate sentence. Don't list them together. Respond with a JSON, with the keys being "1", "2", "3" for each question, and the values being a list. Don't include any other formatting.

Here is a press release: 

PRESS RELEASE: ```{press_release}```
"""


PART_ONE_FEWSHOT_TEMPLATE = """I will show you a press release.
What parts of the press release are "spun", or written to present facts about the company in a positive light, and how might a critical news article de-spin these?

Here are some examples, with abbreviated press releases:

Example 1:
PRESS RELEASE: ```As we have stated many times in the past, Live Nation takes its responsibilities under the antitrust laws seriously and does not engage in behaviors that could justify antitrust litigation, let alone orders that would require it to alter fundamental business practices.```
ANSWER: "The journalist can take a critical angle on the Live Nation Entertainment press release, focusing on the negative aspects of the Ticketmaster-Live Nation merger, including allegations of monopolistic behavior and its impact on ticket prices and market competition."

Example 2:
PRESS RELEASE: ```STAMFORD, Conn - Charter Communications, Inc. (along with its subsidiaries, the "Company" or "Charter") today provided an update on its contract negotiation dispute with The Walt Disney Company. Overview We respect the quality video products that The Walt Disney Company produces as well as the experience of its management team. But the current video ecosystem is broken, and we know there is a better path that will deliver video products with the choice consumers want.```
ANSWER: "The journalist can present the ongoing dispute between Charter Communications and The Walt Disney Company, highlighting the impact on consumers."

Example 3:
PRESS RELEASE: ```Tesla cars come standard with advanced hardware capable of providing Autopilot features, and full self-driving capabilities--through software updates designed to improve functionality over time.```
ANSWER: "The news article can explore controversy surrounding Tesla's marketing of its 'Full Self-Driving' and 'Autopilot' features, inclusing Tesla using misleading language and the potential consequences for Tesla."

Ok, now it's your turn.
PRESS RELEASE: ```{press_release}```

What parts of the press release are "spun", or written to present facts about the company in a positive light, and how might a critical news article de-spin these?
Answer in one sentence. 
ANSWER:
"""


PART_TWO_FEWSHOT_TEMPLATE = """I will show you a press release.
What are potential controversies, or possible story angles I can investigate in this press release? 
Return a python list of sentences. List each controversy in a separate sentence. Don't include any other formatting.

Here are some examples, with abbreviated press releases:

Example 1:
PRESS RELEASE: ```As we have stated many times in the past, Live Nation takes its responsibilities under the antitrust laws seriously and does not engage in behaviors that could justify antitrust litigation, let alone orders that would require it to alter fundamental business practices.```
ANSWER: [
    'The journalist can challenge the press release by highlighting the negative consequences of the Ticketmaster-Live Nation merger, such as high ticket prices and low competition.',
    "The journalist can contrast the press release's claims of competitive behavior and innovation with allegations of a dominant market position leading to less choice and higher ticket fees.",
    'The journalist can emphasize the outrage and disappointment of fans, to illustrate the real-world impact of the issues.',
    "The journalist can question the effectiveness and sincerity of Live Nation's commitment to antitrust laws and consumer-friendly practices."
]

Example 2:
PRESS RELEASE: ```STAMFORD, Conn - Charter Communications, Inc. (along with its subsidiaries, the "Company" or "Charter") today provided an update on its contract negotiation dispute with The Walt Disney Company. Overview We respect the quality video products that The Walt Disney Company produces as well as the experience of its management team. But the current video ecosystem is broken, and we know there is a better path that will deliver video products with the choice consumers want.```
ANSWER: [
    "The article can challenge Charter Communications' portrayal of the dispute by highlighting Disney's actions.",
    "It can contrast Charter's claims by noting Disney's efforts to transition its business models and streamlining its costs.",
    "It can provide a counter-narrative to Charter's assertion of trying to protect consumers from high costs by mentioning Charter's willingness to accept Disney's rate increase under certain conditions.",
    "The report implies that the dispute's impact on consumers extends beyond just costs, touching on how it affects access to content."
]

Example 3:
PRESS RELEASE: ```Tesla cars come standard with advanced hardware capable of providing Autopilot features, and full self-driving capabilities--through software updates designed to improve functionality over time.```
ANSWER: [
    "The article can challenge Tesla's claims by pointing out the discrepancy between Tesla's marketing language and the actual functionality of the 'Autopilot' and 'Full Self-Driving' features, which require active driver supervision.",
    'It can emphasize the seriousness of the misleading marketing by citing the potential legal repercussions Tesla faces, including the possibility of being banned from selling cars in California.',
    "It can highlight skepticism towards Tesla's claims by discussing past scrutiny from national safety organizations and the implications of the company's self-driving technology not being fully autonomous as advertised."
]

Ok, now it's your turn. 
PRESS RELEASE: ```{press_release}"

What are potential controversies, or possible story angles I can investigate in this press release?
Answer in a python list, with each angle being a different list item.
ANSWER:
"""

PART_THREE_FEWSHOT_TEMPLATE = """I will show you a press release. and ask you for some suggestions for a reporter to follow.
Who/what are some sources I should speak to, or resources I should use to pursue these angles?
Return a python list of sentences. List each source in a separate sentence. Don't include any other formatting. 

Here are some examples, with abbreviated press releases:

Example 1:
PRESS RELEASE: ```As we have stated many times in the past, Live Nation takes its responsibilities under the antitrust laws seriously and does not engage in behaviors that could justify antitrust litigation, let alone orders that would require it to alter fundamental business practices.```
ANSWER: [
    'An antitrust expert who testified against the Ticketmaster-Live Nation merger, offering insights into the potential priorities of the DOJ regarding antitrust enforcement.',
    'Customers or fans.',
    'Public lawmakers, including specific members of Congress who have publicly criticized the merger and called for action against the perceived monopoly.',
    "Comparative financial and operational data pre and post-merger, to illustrate the changes in the market dynamics and Live Nation's business performance."
]

Example 2:
PRESS RELEASE: ```STAMFORD, Conn - Charter Communications, Inc. (along with its subsidiaries, the "Company" or "Charter") today provided an update on its contract negotiation dispute with The Walt Disney Company. Overview We respect the quality video products that The Walt Disney Company produces as well as the experience of its management team. But the current video ecosystem is broken, and we know there is a better path that will deliver video products with the choice consumers want.```
ANSWER: [
    'Financial analysis might provide an external perspective on the financial implications of the dispute.',
    "Statements from a Charter Communications spokesperson to represent the company's stance directly.",
    "A blog post from Disney to customers, revealing Disney's side of the story.",
    'Another news source offering a credible journalistic perspective on the issue.'
]

Example 3:
PRESS RELEASE: ```Tesla cars come standard with advanced hardware capable of providing Autopilot features, and full self-driving capabilities--through software updates designed to improve functionality over time.```
ANSWER: [
    'The article can incorporate statements and actions regulatory agencies to frame the scrutiny and potential legal challenges Tesla faces.',
    "Comments and actions by the National Transportation Safety Board (NTSB) are included to discuss past concerns regarding Tesla's feature naming and implications.",
    "The journalist can look at Tesla's website to see if there is a disclaimer, indicating Tesla's acknowledgment of the limitations of Autopilot and Full Self-Driving features which contradicts their descriptions.",
    "An interview with Elon Musk can illustrate the importance placed on self-driving technology by Tesla's CEO.",
    'A DMV spokesperson can be quoted to suggest potential softer regulatory outcomes despite the serious allegations.'
]

Ok, now it's your turn. 
PRESS RELEASE: ```{press_release}"

Who/what are some sources I should speak to, or resources I should use to pursue these angles? 
Answer in a python list of sentences. List each source in a separate sentence.
ANSWER:
"""



PART_ONE_ZEROSHOT_TEMPLATE = """Here is a press release.
PRESS RELEASE: ```{press_release}```

What parts of the press release are "spun", or written to present facts about the company in a positive light, and how might a critical news article de-spin these?
Answer in one sentence. 
ANSWER:
"""


PART_TWO_ZEROSHOT_TEMPLATE = """Here is a press release.
PRESS RELEASE: ```{press_release}```

What are potential controversies, or possible story angles I can investigate in this press release?
Answer in a python list, with each angle being a different list item.
ANSWER:
"""

PART_THREE_ZEROSHOT_TEMPLATE = """Here is a press release.
PRESS RELEASE: ```{press_release}```

Who/what are some sources I should speak to, or resources I should use to pursue these sources? 
Answer in a python list of sentences. List each source in a separate sentence.
ANSWER:
"""



CONSISTENCY_CHECK_PROMPT = """
You will see a {num_pairs} pairs of A. suggestions and B. follow-through. Answer each one with a "Yes" or a "No".
Answer "Yes" if the follow-through is THE SAME as the suggestion. 
Answer "No" if the statements are not consistent and not containing similar information, or if one contains many, many more pieces of information than the other.
Answer each on a new line. DON'T SAY ANYTHING BESIDES "Yes" or "No" and the number of each question you're answering.
Remember that "Yes" means that the information is SUBSTANTIALLY the same between the suggestion and the follow-through.

Here are 6 examples:

1. 
A. A critical news article might de-spin this by questioning the methodology of the safety statistics provided by Tesla and examining independent studies on the effectiveness of Autopilot in preventing accidents.
B. The article challenges the press release's emphasis on Tesla Autopilot's safety statistics by discussing the human aspect of tragedies and the perception that Tesla's responses sometimes lack empathy.
Answer: Yes

2.
A. Potential controversies or story angles could include investigating the accuracy of the company's economic forecasts and market analyses, especially regarding the global recession and GDP projections in different regions.
B. The article highlights the significant decline in exports and industrial production, and the calls by countries and industry leaders for reshoring of critical industries to mitigate vulnerabilities.
Answer: No

3.
A. A critical news article might de-spin these points by examining the challenges faced by Tesla in terms of profitability, product affordability, and competition from established automakers.
B. The news article focuses on the challenges Tesla faces as it aims to remain profitable while making its products more affordable, specifically highlighting the workforce reduction and the upcoming introduction of cheaper Model 3 variants.
Answer: Yes

4.
A. A critical news article could de-spin this by exploring the ethical implications of conducting such surveys in a protected area like the Antarctic, raising questions about environmental impact and adherence to international treaties.
B. It portrays Antarctica as a competitive frontier where major powers, particularly Russia and China, are expanding their influence while Western countries are scaling back due to the COVID-19 pandemic.
Answer: No

5.
A. Speak to current and former Tesla employees to gather firsthand accounts of the company's workplace culture and response to discrimination complaints.
B. Interviews with the victims, providing firsthand accounts of their experiences and the responses (or lack thereof) from Tesla's management and HR.
Answer: Yes

6.
A. Interview regulatory experts or healthcare policymakers to discuss the regulatory process for novel Alzheimer's disease therapies like ADUHELM and the considerations behind accelerated approvals.
B. Congressional investigation findings that reveal Biogen's profit-maximizing motive behind Aduhelm's list price.
Answer: No


Now it's your turn:
{statements}

Now answer. Remember, answer the previous {num_pairs} questions. Answer each one on a new line. Answer with the number of the question and "Yes" or "No". Do not answer anything else. Remember that "Yes" means that the information is SUBSTANTIALLY the same between the suggestion and the follow-through.
"""