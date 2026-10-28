import sqlite3
DB_NAME = "cleanup_requests.db"

def get_connection():
    connection = sqlite3.connect(DB_NAME)
    connection.execute("PRAGMA foreign_keys = ON")
    return connection

def create_table():
    connection = get_connection()

    # Existing waste reports table.
    connection.execute("""
        CREATE TABLE IF NOT EXISTS requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            location TEXT NOT NULL,
            description TEXT,
            category TEXT NOT NULL,
            confidence REAL NOT NULL,
            image BLOB NOT NULL,
            status TEXT DEFAULT 'Pending'
        )
    """)

    # Register cleanup teams and their service areas.
    connection.execute("""
        CREATE TABLE IF NOT EXISTS cleanup_teams (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            team_name TEXT NOT NULL,
            service_area TEXT NOT NULL,
            base_lat REAL,
            base_lon REAL,
            mileage_kmpl REAL DEFAULT 15,
            fuel_price REAL DEFAULT 100,
            is_active INTEGER DEFAULT 1
        )
    """)

    # Add new fields to an existing requests table safely.
    existing_columns = {
        row[1]
        for row in connection.execute(
            "PRAGMA table_info(requests)"
        ).fetchall()
    }

    new_columns = {
        "assigned_team_id": "INTEGER",
        "estimated_distance_km": "REAL",
        "estimated_fuel_cost": "REAL",
        "assignment_status": "TEXT DEFAULT 'Unassigned'",
    }

    for column_name, column_type in new_columns.items():
        if column_name not in existing_columns:
            connection.execute(
                f"ALTER TABLE requests "
                f"ADD COLUMN {column_name} {column_type}"
            )

    connection.commit()
    connection.close()

def add_request(location, description, category, confidence, image):
    connection = get_connection()

    connection.execute("""
        INSERT INTO requests
        (location, description, category, confidence, image, status)
        VALUES (?, ?, ?, ?, ?, 'Pending')
    """, (
        location,
        description,
        category,
        confidence,
        image,
    ))

    connection.commit()
    connection.close()

def get_requests():
    connection = get_connection()

    cursor = connection.execute("""
        SELECT id, location, description,
               category, confidence, status
        FROM requests
        ORDER BY id DESC
    """)

    requests = cursor.fetchall()
    connection.close()

    return requests

def get_request_image(request_id):
    connection = get_connection()

    cursor = connection.execute(
        "SELECT image FROM requests WHERE id = ?",
        (request_id,)
    )

    result = cursor.fetchone()
    connection.close()

    return result[0] if result else None

def update_status(request_id, status):
    connection = get_connection()

    connection.execute(
        "UPDATE requests SET status = ? WHERE id = ?",
        (status, request_id)
    )

    connection.commit()
    connection.close()

def add_cleanup_team(
    team_name,
    service_area,
    base_lat=None,
    base_lon=None,
    mileage_kmpl=15,
    fuel_price=100
):
    """Register a cleanup team."""

    if not team_name.strip() or not service_area.strip():
        raise ValueError("Team name and service area are required.")

    if mileage_kmpl <= 0 or fuel_price < 0:
        raise ValueError("Mileage must be positive and fuel price cannot be negative.")

    connection = get_connection()

    cursor = connection.execute("""
        INSERT INTO cleanup_teams
        (team_name, service_area, base_lat, base_lon,
         mileage_kmpl, fuel_price, is_active)
        VALUES (?, ?, ?, ?, ?, ?, 1)
    """, (
        team_name.strip(),
        service_area.strip(),
        base_lat,
        base_lon,
        mileage_kmpl,
        fuel_price,
    ))

    team_id = cursor.lastrowid

    connection.commit()
    connection.close()

    return team_id

def get_cleanup_teams(active_only=True):
    """Return registered cleanup teams."""

    connection = get_connection()

    if active_only:
        cursor = connection.execute("""
            SELECT id, team_name, service_area,
                   base_lat, base_lon,
                   mileage_kmpl, fuel_price, is_active
            FROM cleanup_teams
            WHERE is_active = 1
            ORDER BY team_name
        """)
    else:
        cursor = connection.execute("""
            SELECT id, team_name, service_area,
                   base_lat, base_lon,
                   mileage_kmpl, fuel_price, is_active
            FROM cleanup_teams
            ORDER BY team_name
        """)

    teams = cursor.fetchall()
    connection.close()

    return teams

def set_team_active(team_id, is_active):
    """Activate or deactivate a cleanup team."""

    connection = get_connection()

    connection.execute("""
        UPDATE cleanup_teams
        SET is_active = ?
        WHERE id = ?
    """, (1 if is_active else 0, team_id))

    connection.commit()
    connection.close()

def assign_cleanup_team(
    request_id,
    team_id,
    distance_km,
    estimated_fuel_cost
):
    """Assign a team and save estimated travel expenses."""

    if distance_km < 0 or estimated_fuel_cost < 0:
        raise ValueError("Distance and estimated cost cannot be negative.")

    connection = get_connection()

    team = connection.execute("""
        SELECT id
        FROM cleanup_teams
        WHERE id = ? AND is_active = 1
    """, (team_id,)).fetchone()

    request_exists = connection.execute("""
        SELECT id
        FROM requests
        WHERE id = ?
    """, (request_id,)).fetchone()

    if not team:
        connection.close()
        raise ValueError("The selected team does not exist or is inactive.")

    if not request_exists:
        connection.close()
        raise ValueError("The selected cleanup request does not exist.")

    connection.execute("""
        UPDATE requests
        SET assigned_team_id = ?,
            estimated_distance_km = ?,
            estimated_fuel_cost = ?,
            assignment_status = 'Assigned'
        WHERE id = ?
    """, (
        team_id,
        distance_km,
        estimated_fuel_cost,
        request_id,
    ))

    connection.commit()
    connection.close()

def get_assignment_details(request_id):
    """Return assignment information for a cleanup request."""

    connection = get_connection()

    cursor = connection.execute("""
        SELECT
            r.id,
            r.assigned_team_id,
            t.team_name,
            t.service_area,
            r.estimated_distance_km,
            r.estimated_fuel_cost,
            r.assignment_status
        FROM requests AS r
        LEFT JOIN cleanup_teams AS t
            ON r.assigned_team_id = t.id
        WHERE r.id = ?
    """, (request_id,))

    result = cursor.fetchone()
    connection.close()
    return result

def update_assignment_status(request_id, assignment_status):
    """Update the assignment workflow status."""

    allowed_statuses = {
        "Unassigned",
        "Pending Assignment",
        "Assigned",
        "In Progress",
        "Completed",
    }

    if assignment_status not in allowed_statuses:
        raise ValueError("Invalid assignment status.")

    connection = get_connection()

    connection.execute("""
        UPDATE requests
        SET assignment_status = ?
        WHERE id = ?
    """, (assignment_status, request_id))

    connection.commit()
    connection.close()
# Create or upgrade tables when this module is imported.
create_table()