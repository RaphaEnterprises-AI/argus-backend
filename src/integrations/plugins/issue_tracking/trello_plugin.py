"""Trello integration plugin.

Trello REST API with API Key + Token authentication.
API Documentation: https://developer.atlassian.com/cloud/trello/rest/
"""

import time
from datetime import datetime
from typing import Any, Optional

import httpx

from src.integrations.base import (
    AuthType,
    ConnectionTestResult,
    FieldDefinition,
    IntegrationCategory,
    IntegrationCredentials,
    IntegrationMetadata,
    IntegrationPlugin,
    SyncResult,
)


class TrelloPlugin(IntegrationPlugin):
    """Trello integration for visual project management with boards, lists, and cards.

    Trello uses a Kanban-style approach to project management with boards,
    lists, and cards for organizing work.

    Uses API Key + Token authentication.
    API Documentation: https://developer.atlassian.com/cloud/trello/rest/
    """

    BASE_URL = "https://api.trello.com/1"

    @property
    def metadata(self) -> IntegrationMetadata:
        """Return Trello integration metadata."""
        return IntegrationMetadata(
            slug="trello",
            name="Trello",
            description="Connect to Trello for visual project management with boards, lists, and cards. Organize work in a Kanban-style interface.",
            category=IntegrationCategory.ISSUE_TRACKING,
            auth_type=AuthType.API_KEY,
            icon_url="https://cdn.worldvectorlogo.com/logos/trello.svg",
            docs_url="https://developer.atlassian.com/cloud/trello/rest/",
            website_url="https://trello.com",
            features=[
                "boards",
                "lists",
                "cards",
                "checklists",
                "labels",
                "members",
                "attachments",
                "comments",
                "power_ups",
                "webhooks",
            ],
            required_fields=[
                FieldDefinition(
                    name="api_key",
                    label="API Key",
                    type="password",
                    required=True,
                    placeholder="32-character key",
                    help_text="Get your API key at https://trello.com/app-key",
                ),
                FieldDefinition(
                    name="access_token",
                    label="Token",
                    type="password",
                    required=True,
                    placeholder="64-character token",
                    help_text="Generate a token using your API key at https://trello.com/app-key",
                ),
            ],
            optional_fields=[
                FieldDefinition(
                    name="org_slug",
                    label="Organization ID/Name",
                    type="text",
                    required=False,
                    placeholder="myorganization",
                    help_text="Optional: Limit sync to a specific organization",
                ),
            ],
            supports_webhooks=True,
            supports_sync=True,
            sync_resources=["boards", "lists", "cards", "labels", "members", "checklists"],
        )

    def _get_auth_params(self, credentials: IntegrationCredentials) -> dict[str, str]:
        """Get authentication query parameters."""
        return {
            "key": credentials.api_key,
            "token": credentials.access_token,
        }

    async def test_connection(
        self,
        credentials: IntegrationCredentials,
    ) -> ConnectionTestResult:
        """Test connection to Trello API.

        Uses GET /members/me to verify credentials.
        """
        start_time = time.time()

        try:
            params = self._get_auth_params(credentials)

            # Test connection with /members/me endpoint
            response = await self.http_client.get(
                f"{self.BASE_URL}/members/me",
                params={
                    **params,
                    "fields": "id,fullName,username,email,idOrganizations,url",
                    "organizations": "all",
                    "organization_fields": "id,displayName,name",
                },
            )

            latency_ms = (time.time() - start_time) * 1000

            if response.status_code == 200:
                user_data = response.json()

                # Get boards count for permissions context
                boards_response = await self.http_client.get(
                    f"{self.BASE_URL}/members/me/boards",
                    params={**params, "fields": "id"},
                )
                boards_count = 0
                if boards_response.status_code == 200:
                    boards_count = len(boards_response.json())

                organizations = user_data.get("organizations", [])
                permissions = [
                    "read:members",
                    "read:boards",
                    "read:cards",
                    "read:lists",
                    f"access:{boards_count}_boards",
                    f"access:{len(organizations)}_organizations",
                ]

                return ConnectionTestResult(
                    success=True,
                    message=f"Connected as {user_data.get('fullName', user_data.get('username', 'Unknown'))}",
                    latency_ms=latency_ms,
                    api_version="1",
                    permissions=permissions,
                    account_info={
                        "member_id": user_data.get("id"),
                        "username": user_data.get("username"),
                        "full_name": user_data.get("fullName"),
                        "email": user_data.get("email"),
                        "url": user_data.get("url"),
                        "organizations": [
                            {"id": org.get("id"), "name": org.get("displayName", org.get("name"))}
                            for org in organizations[:10]
                        ],
                        "boards_count": boards_count,
                    },
                )

            elif response.status_code == 401:
                return ConnectionTestResult(
                    success=False,
                    message="Authentication failed. Check your API key and token.",
                    latency_ms=latency_ms,
                    error_code="UNAUTHORIZED",
                    error_details={"status_code": 401},
                )
            elif response.status_code == 403:
                return ConnectionTestResult(
                    success=False,
                    message="Access forbidden. Your token may have expired or lacks permissions.",
                    latency_ms=latency_ms,
                    error_code="FORBIDDEN",
                    error_details={"status_code": 403},
                )
            else:
                return ConnectionTestResult(
                    success=False,
                    message=f"Unexpected response: {response.status_code}",
                    latency_ms=latency_ms,
                    error_code="UNEXPECTED_ERROR",
                    error_details={
                        "status_code": response.status_code,
                        "response": response.text[:500],
                    },
                )

        except httpx.TimeoutException:
            return ConnectionTestResult(
                success=False,
                message="Connection timed out",
                error_code="TIMEOUT",
            )
        except httpx.ConnectError as e:
            return ConnectionTestResult(
                success=False,
                message=f"Failed to connect: {str(e)}",
                error_code="CONNECTION_ERROR",
            )
        except Exception as e:
            return ConnectionTestResult(
                success=False,
                message=f"Connection test failed: {str(e)}",
                error_code="UNKNOWN_ERROR",
                error_details={"exception": str(e)},
            )

    async def sync_data(
        self,
        credentials: IntegrationCredentials,
        resources: Optional[list[str]] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> SyncResult:
        """Sync data from Trello.

        Syncs boards, lists, and cards.
        """
        started_at = datetime.utcnow()
        sync_id = f"trello_sync_{int(started_at.timestamp())}"

        data: dict[str, Any] = {}
        errors: list[str] = []
        warnings: list[str] = []
        total_items = 0

        # Default to all resources if none specified
        resources_to_sync = resources or ["boards", "lists", "cards"]

        try:
            params = self._get_auth_params(credentials)

            # Sync boards
            boards = []
            if "boards" in resources_to_sync:
                try:
                    # Build board filter
                    filter_param = "all"  # open, closed, members, organization, public, starred

                    boards_response = await self.http_client.get(
                        f"{self.BASE_URL}/members/me/boards",
                        params={
                            **params,
                            "filter": filter_param,
                            "fields": "id,name,desc,closed,url,shortUrl,dateLastActivity,idOrganization,starred,memberships",
                            "lists": "open",
                            "list_fields": "id,name,closed,pos",
                            "organization": "true",
                            "organization_fields": "id,displayName",
                        },
                    )

                    if boards_response.status_code == 200:
                        boards = boards_response.json()

                        # Filter by organization if specified
                        if credentials.org_slug:
                            boards = [b for b in boards if b.get("idOrganization") == credentials.org_slug]

                        # Apply limit
                        boards = boards[:limit]

                        # Filter by since if provided (Trello uses dateLastActivity)
                        if since:
                            since_str = since.isoformat()
                            boards = [
                                b for b in boards
                                if b.get("dateLastActivity") and b.get("dateLastActivity") >= since_str
                            ]

                        data["boards"] = [
                            {
                                "id": b.get("id"),
                                "name": b.get("name"),
                                "description": b.get("desc", "")[:500] if b.get("desc") else None,
                                "closed": b.get("closed"),
                                "url": b.get("url"),
                                "short_url": b.get("shortUrl"),
                                "last_activity": b.get("dateLastActivity"),
                                "organization_id": b.get("idOrganization"),
                                "organization_name": b.get("organization", {}).get("displayName") if b.get("organization") else None,
                                "starred": b.get("starred"),
                                "lists_count": len(b.get("lists", [])),
                            }
                            for b in boards
                        ]
                        total_items += len(boards)
                    else:
                        errors.append(f"Failed to fetch boards: {boards_response.status_code}")
                except Exception as e:
                    errors.append(f"Error fetching boards: {str(e)}")

            # Sync lists
            if "lists" in resources_to_sync:
                all_lists = []
                for board in boards[:10]:  # Limit to first 10 boards
                    board_lists = board.get("lists", [])
                    for lst in board_lists:
                        all_lists.append({
                            "id": lst.get("id"),
                            "name": lst.get("name"),
                            "closed": lst.get("closed"),
                            "position": lst.get("pos"),
                            "board_id": board.get("id"),
                            "board_name": board.get("name"),
                        })

                data["lists"] = all_lists
                total_items += len(all_lists)

            # Sync cards
            if "cards" in resources_to_sync:
                all_cards = []
                for board in boards[:10]:  # Limit to first 10 boards
                    try:
                        cards_params: dict[str, Any] = {
                            **params,
                            "fields": "id,name,desc,closed,url,shortUrl,due,dueComplete,dateLastActivity,idList,idMembers,labels,badges",
                            "members": "true",
                            "member_fields": "id,fullName,username",
                            "checklists": "all" if "checklists" in resources_to_sync else "none",
                            "limit": min(limit // len(boards) if boards else limit, 100),
                        }

                        # Trello doesn't have a direct "since" filter for cards, but we can filter after
                        cards_response = await self.http_client.get(
                            f"{self.BASE_URL}/boards/{board.get('id')}/cards",
                            params=cards_params,
                        )

                        if cards_response.status_code == 200:
                            cards = cards_response.json()

                            # Filter by since if provided
                            if since:
                                since_str = since.isoformat()
                                cards = [
                                    c for c in cards
                                    if c.get("dateLastActivity") and c.get("dateLastActivity") >= since_str
                                ]

                            for card in cards:
                                card_data = {
                                    "id": card.get("id"),
                                    "name": card.get("name"),
                                    "description": card.get("desc", "")[:500] if card.get("desc") else None,
                                    "closed": card.get("closed"),
                                    "url": card.get("url"),
                                    "short_url": card.get("shortUrl"),
                                    "due": card.get("due"),
                                    "due_complete": card.get("dueComplete"),
                                    "last_activity": card.get("dateLastActivity"),
                                    "list_id": card.get("idList"),
                                    "board_id": board.get("id"),
                                    "board_name": board.get("name"),
                                    "members": [
                                        {"id": m.get("id"), "name": m.get("fullName"), "username": m.get("username")}
                                        for m in card.get("members", [])
                                    ],
                                    "labels": [
                                        {"id": l.get("id"), "name": l.get("name"), "color": l.get("color")}
                                        for l in card.get("labels", [])
                                    ],
                                    "badges": {
                                        "comments": card.get("badges", {}).get("comments", 0),
                                        "attachments": card.get("badges", {}).get("attachments", 0),
                                        "checkitems": card.get("badges", {}).get("checkItems", 0),
                                        "checkitems_checked": card.get("badges", {}).get("checkItemsChecked", 0),
                                    },
                                }

                                # Include checklists if synced
                                if "checklists" in resources_to_sync and card.get("checklists"):
                                    card_data["checklists"] = [
                                        {
                                            "id": cl.get("id"),
                                            "name": cl.get("name"),
                                            "items": [
                                                {
                                                    "id": item.get("id"),
                                                    "name": item.get("name"),
                                                    "state": item.get("state"),
                                                }
                                                for item in cl.get("checkItems", [])
                                            ],
                                        }
                                        for cl in card.get("checklists", [])
                                    ]

                                all_cards.append(card_data)
                    except Exception as e:
                        warnings.append(f"Error fetching cards for board {board.get('id')}: {str(e)}")

                data["cards"] = all_cards
                total_items += len(all_cards)

            # Sync labels (board-level)
            if "labels" in resources_to_sync:
                all_labels = []
                for board in boards[:10]:
                    try:
                        labels_response = await self.http_client.get(
                            f"{self.BASE_URL}/boards/{board.get('id')}/labels",
                            params={**params, "fields": "id,name,color"},
                        )

                        if labels_response.status_code == 200:
                            labels = labels_response.json()
                            for label in labels:
                                if label.get("name"):  # Skip unnamed labels
                                    all_labels.append({
                                        "id": label.get("id"),
                                        "name": label.get("name"),
                                        "color": label.get("color"),
                                        "board_id": board.get("id"),
                                        "board_name": board.get("name"),
                                    })
                    except Exception as e:
                        warnings.append(f"Error fetching labels for board {board.get('id')}: {str(e)}")

                data["labels"] = all_labels
                total_items += len(all_labels)

            # Sync members
            if "members" in resources_to_sync:
                all_members = []
                seen_members = set()
                for board in boards[:10]:
                    try:
                        members_response = await self.http_client.get(
                            f"{self.BASE_URL}/boards/{board.get('id')}/members",
                            params={
                                **params,
                                "fields": "id,fullName,username,avatarUrl",
                            },
                        )

                        if members_response.status_code == 200:
                            members = members_response.json()
                            for member in members:
                                if member.get("id") not in seen_members:
                                    seen_members.add(member.get("id"))
                                    all_members.append({
                                        "id": member.get("id"),
                                        "full_name": member.get("fullName"),
                                        "username": member.get("username"),
                                        "avatar_url": member.get("avatarUrl"),
                                    })
                    except Exception as e:
                        warnings.append(f"Error fetching members for board {board.get('id')}: {str(e)}")

                data["members"] = all_members
                total_items += len(all_members)

            completed_at = datetime.utcnow()

            return SyncResult(
                success=len(errors) == 0,
                message=f"Synced {total_items} items from Trello" if not errors else f"Partial sync: {total_items} items with {len(errors)} errors",
                data_points_synced=total_items,
                sync_id=sync_id,
                started_at=started_at,
                completed_at=completed_at,
                errors=errors,
                warnings=warnings,
                data=data,
            )

        except Exception as e:
            return SyncResult(
                success=False,
                message=f"Sync failed: {str(e)}",
                errors=[str(e)],
                sync_id=sync_id,
                started_at=started_at,
                completed_at=datetime.utcnow(),
            )
