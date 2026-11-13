"""PII Anonymizer and Redactor for WhatsApp chats.
Masks real names, phone numbers, email addresses, and form/token links before
sending messages to any external or local AI model, with optional local de-anonymization.
"""
import json
import re
from dataclasses import replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from parser import Message


class Anonymizer:
    def __init__(
        self,
        mask_names: bool = True,
        mask_phones: bool = True,
        mask_links: bool = True,
        mask_emails: bool = True,
    ):
        self.mask_names = mask_names
        self.mask_phones = mask_phones
        self.mask_links = mask_links
        self.mask_emails = mask_emails
        self.sender_map: dict[str, str] = {}
        self.reverse_map: dict[str, str] = {}
        self._counter = 1

    def get_alias(self, real_name: str) -> str:
        """Get or create a consistent pseudonym for a participant."""
        if real_name not in self.sender_map:
            alias = f"Participant_{self._counter}"
            self.sender_map[real_name] = alias
            self.reverse_map[alias] = real_name
            self._counter += 1
        return self.sender_map[real_name]

    def anonymize_text(self, text: str) -> str:
        """Scrub phone numbers, emails, links, and participant names from text."""
        # 1. Emails
        if self.mask_emails:
            text = re.sub(
                r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
                "[EMAIL_REDACTED]",
                text,
            )

        # 2. Phone numbers (e.g. +91 90341 54114, 080-1234567, +1 (555) 019-2834)
        if self.mask_phones:
            text = re.sub(
                r"(\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,5}[-.\s]?\d{3,5}",
                "[PHONE_REDACTED]",
                text,
            )

        # 3. URLs, forms, and tokens (protects Google Forms tokens, meeting links, etc.)
        if self.mask_links:
            text = re.sub(r"https?://[^\s]+", "[LINK_REDACTED]", text)

        # 4. Known participant names mentioned in conversation text
        if self.mask_names and self.sender_map:
            # Sort by descending length so "Akshat Varshney" is replaced before "Akshat"
            for real_name, alias in sorted(
                self.sender_map.items(), key=lambda x: len(x[0]), reverse=True
            ):
                if len(real_name) >= 3 and not real_name.startswith("+"):
                    text = re.sub(
                        r"\b" + re.escape(real_name) + r"\b",
                        alias,
                        text,
                        flags=re.I,
                    )

        return text

    def anonymize_messages(self, messages: list["Message"]) -> list["Message"]:
        """Anonymize a list of Message objects."""
        # Pre-populate sender aliases for all participants
        if self.mask_names:
            for m in messages:
                self.get_alias(m.sender)

        anonymized = []
        for m in messages:
            sender = self.get_alias(m.sender) if self.mask_names else m.sender
            text = self.anonymize_text(m.text)
            anonymized.append(replace(m, sender=sender, text=text))

        return anonymized

    def de_anonymize_result(self, result: dict) -> dict:
        """Restore real participant names locally in the parsed summary dictionary."""
        if not self.reverse_map:
            return result

        try:
            dumped = json.dumps(result, ensure_ascii=False)
            # Replace aliases with real names (sort by alias length descending)
            for alias, real_name in sorted(
                self.reverse_map.items(), key=lambda x: len(x[0]), reverse=True
            ):
                dumped = dumped.replace(alias, real_name)
            return json.loads(dumped)
        except Exception:
            return result

