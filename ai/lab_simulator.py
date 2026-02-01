"""
Lab Simulator: Creates fake environments for stress testing.
Simulates inboxes, calendars, files, devices without real data.
"""

import json
import os
import random
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import uuid


class LabSimulator:
    """Simulates fake environments for ALICE testing."""

    def __init__(self, output_dir: str = "data/lab/"):
        """Initialize lab simulator."""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    # ===== FAKE INBOX =====
    def generate_fake_emails(self, count: int = 50) -> List[Dict[str, Any]]:
        """Generate realistic fake emails."""
        senders = ["boss@company.com", "john@example.com", "jane@example.com",
                  "newsletter@news.com", "support@service.com"]
        subjects = [
            "Meeting tomorrow at 2pm",
            "Project status update",
            "Action required: Review document",
            "Your subscription confirmation",
            "Weekly digest",
            "Team lunch reminder",
            "Bug fix deployed",
            "Client feedback",
        ]

        emails = []
        base_time = datetime.now()

        for i in range(count):
            email = {
                "id": str(uuid.uuid4()),
                "from": random.choice(senders),
                "subject": random.choice(subjects),
                "body": f"Email body {i}. Lorem ipsum dolor sit amet...",
                "timestamp": (base_time - timedelta(hours=random.randint(0, 720))).isoformat(),
                "read": random.choice([True, False]),
                "starred": random.choice([True, False, False, False])
            }
            emails.append(email)

        return sorted(emails, key=lambda x: x['timestamp'], reverse=True)

    def save_fake_inbox(self, count: int = 50) -> str:
        """Save fake inbox to file."""
        emails = self.generate_fake_emails(count)
        inbox_path = os.path.join(self.output_dir, "fake_inbox.json")
        with open(inbox_path, 'w') as f:
            json.dump(emails, f, indent=2)
        return inbox_path

    # ===== FAKE CALENDAR =====
    def generate_fake_calendar_events(self, days: int = 30) -> List[Dict[str, Any]]:
        """Generate realistic fake calendar events."""
        event_types = ["meeting", "standup", "one-on-one", "lunch", "conference",
                      "training", "retrospective", "planning"]

        events = []
        base_date = datetime.now()

        for day_offset in range(days):
            current_date = base_date + timedelta(days=day_offset)

            # 2-4 events per day
            num_events = random.randint(2, 4)
            for _ in range(num_events):
                hour = random.randint(8, 17)
                duration = random.choice([30, 60, 90])

                event = {
                    "id": str(uuid.uuid4()),
                    "title": f"{random.choice(event_types).title()} - {random.choice(['Team', 'Client', 'Internal'])}",
                    "start": current_date.replace(hour=hour, minute=0, second=0).isoformat(),
                    "end": (current_date.replace(hour=hour, minute=0, second=0) + timedelta(minutes=duration)).isoformat(),
                    "location": random.choice(["Conference Room A", "Zoom", "My Office", "Cafeteria"]),
                    "attendees": random.randint(1, 8),
                    "description": "Meeting description placeholder"
                }
                events.append(event)

        return sorted(events, key=lambda x: x['start'])

    def save_fake_calendar(self, days: int = 30) -> str:
        """Save fake calendar to file."""
        events = self.generate_fake_calendar_events(days)
        calendar_path = os.path.join(self.output_dir, "fake_calendar.json")
        with open(calendar_path, 'w') as f:
            json.dump(events, f, indent=2)
        return calendar_path

    # ===== FAKE FILE SYSTEM =====
    def generate_fake_files(self, count: int = 100) -> List[Dict[str, Any]]:
        """Generate realistic fake file metadata."""
        file_types = {
            ".py": "Python",
            ".js": "JavaScript",
            ".txt": "Text",
            ".json": "JSON",
            ".md": "Markdown",
            ".pdf": "PDF",
            ".xlsx": "Excel",
            ".docx": "Word"
        }

        directories = ["Documents", "Projects", "Work", "Personal", "Archive"]
        files = []

        for i in range(count):
            ext = random.choice(list(file_types.keys()))
            file_obj = {
                "id": str(uuid.uuid4()),
                "name": f"file_{i}{ext}",
                "path": f"/{random.choice(directories)}/",
                "type": file_types[ext],
                "size_kb": random.randint(1, 5000),
                "created": (datetime.now() - timedelta(days=random.randint(0, 365))).isoformat(),
                "modified": (datetime.now() - timedelta(days=random.randint(0, 100))).isoformat(),
                "accessed": (datetime.now() - timedelta(hours=random.randint(0, 168))).isoformat()
            }
            files.append(file_obj)

        return sorted(files, key=lambda x: x['modified'], reverse=True)

    def save_fake_files(self, count: int = 100) -> str:
        """Save fake files to file."""
        files = self.generate_fake_files(count)
        files_path = os.path.join(self.output_dir, "fake_files.json")
        with open(files_path, 'w') as f:
            json.dump(files, f, indent=2)
        return files_path

    # ===== FAKE DEVICE STATE =====
    def generate_fake_device_state(self) -> Dict[str, Any]:
        """Generate fake device/system state."""
        return {
            "device": {
                "name": "DESKTOP-ABC123",
                "os": "Windows 11",
                "hostname": "gabriel-device"
            },
            "battery": {
                "percent": random.randint(20, 100),
                "status": random.choice(["charging", "discharging", "full"])
            },
            "network": {
                "connected": True,
                "ssid": "HomeNetwork",
                "signal_strength": random.randint(40, 100),
                "ip_address": f"192.168.1.{random.randint(100, 200)}"
            },
            "storage": {
                "total_gb": 500,
                "used_gb": random.randint(100, 400),
                "free_gb": random.randint(50, 300)
            },
            "applications": {
                "running": ["VS Code", "Chrome", "Slack", "Teams"],
                "updates_available": random.randint(0, 5)
            },
            "timestamp": datetime.now().isoformat()
        }

    def save_fake_device_state(self) -> str:
        """Save fake device state to file."""
        state = self.generate_fake_device_state()
        state_path = os.path.join(self.output_dir, "fake_device_state.json")
        with open(state_path, 'w') as f:
            json.dump(state, f, indent=2)
        return state_path

    # ===== SCENARIO GENERATION =====
    def generate_scenario(self, scenario_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a complete test scenario.
        Combines inbox, calendar, files, device state.
        """
        if scenario_id is None:
            scenario_id = str(uuid.uuid4())[:8]

        scenario = {
            "id": scenario_id,
            "created": datetime.now().isoformat(),
            "inbox": self.generate_fake_emails(random.randint(20, 100)),
            "calendar": self.generate_fake_calendar_events(random.randint(10, 30)),
            "files": self.generate_fake_files(random.randint(50, 200)),
            "device_state": self.generate_fake_device_state()
        }

        return scenario

    def save_scenario(self, scenario: Dict[str, Any]) -> str:
        """Save complete scenario to file."""
        scenario_path = os.path.join(self.output_dir, f"scenario_{scenario['id']}.json")
        with open(scenario_path, 'w') as f:
            json.dump(scenario, f, indent=2)
        return scenario_path

    def generate_batch_scenarios(self, count: int = 10) -> List[str]:
        """Generate batch of N test scenarios."""
        paths = []
        for i in range(count):
            scenario = self.generate_scenario(f"scenario_{i}")
            path = self.save_scenario(scenario)
            paths.append(path)
        return paths

    # ===== STRESS TEST =====
    def generate_stress_test_queries(self, scenario: Dict[str, Any], 
                                    num_queries: int = 50) -> List[Dict[str, Any]]:
        """Generate N realistic queries against a scenario."""
        queries = []

        inbox = scenario.get("inbox", [])
        calendar = scenario.get("calendar", [])
        files = scenario.get("files", [])

        query_templates = [
            # Email queries
            lambda: {"query": "Show me my unread emails", "type": "email", "expected_tool": "email"},
            lambda: {"query": f"Any emails from {inbox[0].get('from', 'someone')} if inbox else 'sender'?", "type": "email", "expected_tool": "email"},
            lambda: {"query": "How many emails do I have", "type": "email_count", "expected_tool": "email"},

            # Calendar queries
            lambda: {"query": "What's my schedule today", "type": "calendar", "expected_tool": "calendar"},
            lambda: {"query": f"Do I have any meetings with 'Client'", "type": "calendar_filter", "expected_tool": "calendar"},
            lambda: {"query": "Show me next week's events", "type": "calendar", "expected_tool": "calendar"},

            # File queries
            lambda: {"query": "List my recent files", "type": "file_browse", "expected_tool": "file_operations"},
            lambda: {"query": "What Python files do I have", "type": "file_filter", "expected_tool": "file_operations"},
            lambda: {"query": "Show me files from Documents", "type": "file_directory", "expected_tool": "file_operations"},

            # General queries
            lambda: {"query": "How much disk space do I have", "type": "system", "expected_tool": "system"},
            lambda: {"query": "What's my device status", "type": "device", "expected_tool": "system"},
        ]

        for _ in range(num_queries):
            query_func = random.choice(query_templates)
            query = query_func()
            query["timestamp"] = datetime.now().isoformat()
            query["id"] = str(uuid.uuid4())[:8]
            queries.append(query)

        return queries

    def save_stress_test(self, scenario: Dict[str, Any], 
                        num_queries: int = 50) -> str:
        """Generate and save stress test."""
        queries = self.generate_stress_test_queries(scenario, num_queries)
        test_path = os.path.join(self.output_dir, f"stress_test_{scenario['id']}.jsonl")
        
        with open(test_path, 'w') as f:
            for query in queries:
                f.write(json.dumps(query) + '\n')

        return test_path

    def get_lab_statistics(self) -> Dict[str, Any]:
        """Get statistics about generated lab data."""
        return {
            "output_dir": self.output_dir,
            "lab_files": [f for f in os.listdir(self.output_dir) if os.path.isfile(os.path.join(self.output_dir, f))],
            "total_size_mb": sum(os.path.getsize(os.path.join(self.output_dir, f)) 
                                for f in os.listdir(self.output_dir)) / (1024 * 1024)
        }
